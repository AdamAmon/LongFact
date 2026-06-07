#!/usr/bin/env bash
# ============================================================================
# LongFact 多 GPU 并行启动脚本（方案 A：数据分片）
#
# 用法：
#   bash scripts/launch_multi_gpu.sh [GPU_COUNT] [TOTAL_N]
#
# 示例（8 张 T4，500 样本）：
#   bash scripts/launch_multi_gpu.sh 8 500
#
# 每张 GPU 独立运行一个进程，处理不同片段的样本，最终自动合并结果。
# ============================================================================
set -euo pipefail

# ── 参数 ──────────────────────────────────────────────────────────────────
GPU_COUNT="${1:-8}"                          # GPU 数量
TOTAL_N="${2:-500}"                           # 总样本数
BATCH_SIZE="${3:-32}"                         # summary_batch_size
MAX_NEW_TOKENS="${4:-256}"                    # summary_max_new_tokens
RETRIEVAL="${5:-dce}"                         # 检索策略：baseline | dce
PRECISION="${6:-fp16}"                        # 精度：fp16 | 8bit

OUT_DIR="results/multi_gpu_${GPU_COUNT}gpu"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUT_DIR"

# ── 计算分片 ──────────────────────────────────────────────────────────────
PER_GPU=$(( (TOTAL_N + GPU_COUNT - 1) / GPU_COUNT ))
echo "============================================================"
echo " LongFact 多 GPU 并行启动"
echo "============================================================"
echo " GPU 数量     : ${GPU_COUNT}"
echo " 总样本数    : ${TOTAL_N}"
echo " 每 GPU 约   : ${PER_GPU} 样本"
echo " batch_size  : ${BATCH_SIZE}"
echo " max_tokens  : ${MAX_NEW_TOKENS}"
echo " 检索策略    : ${RETRIEVAL}"
echo " 精度        : ${PRECISION}"
echo " 输出目录    : ${OUT_DIR}"
echo " 启动时间    : ${TIMESTAMP}"
echo "============================================================"

# ── 清理旧结果（可选，避免与之前运行冲突） ───────────────────────────────
# 如果想保留旧结果请注释掉下面这行
rm -f "${OUT_DIR}"/gpu*.jsonl "${OUT_DIR}"/gpu*.log

# ── 启动各 GPU 进程 ───────────────────────────────────────────────────────
PIDS=()
FAIL_FLAG=0

for ((gpu=0; gpu<GPU_COUNT; gpu++)); do
    START=$(( gpu * PER_GPU ))
    N=$PER_GPU
    # 最后一张 GPU 只处理剩余样本
    if (( gpu == GPU_COUNT - 1 )); then
        N=$(( TOTAL_N - START ))
    fi

    OUTFILE="${OUT_DIR}/gpu${gpu}_n${N}.jsonl"
    LOGFILE="${OUT_DIR}/gpu${gpu}.log"

    echo ""
    echo "[GPU ${gpu}] 启动: start=${START} n=${N} → ${OUTFILE}"
    echo "  日志: ${LOGFILE}"

    CUDA_VISIBLE_DEVICES=${gpu} \
        python run_experiment.py \
            --n "${N}" \
            --start "${START}" \
            --use_model \
            --device 0 \
            --precision "${PRECISION}" \
            --summary_batch_size "${BATCH_SIZE}" \
            --summary_max_new_tokens "${MAX_NEW_TOKENS}" \
            --retrieval_strategy "${RETRIEVAL}" \
            --out "${OUTFILE}" \
        > "${LOGFILE}" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "============================================================"
echo " 所有 ${GPU_COUNT} 个进程已启动，等待完成..."
echo " 监控: watch -n 2 'ps aux | grep run_experiment | wc -l'"
echo " 显存: watch -n 1 nvidia-smi"
echo "============================================================"

# ── 等待完成 ──────────────────────────────────────────────────────────────
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    if wait "$pid"; then
        echo "[GPU ${i}] ✅ 完成 (pid=${pid})"
    else
        echo "[GPU ${i}] ❌ 失败 (pid=${pid})，检查 ${OUT_DIR}/gpu${i}.log"
        FAIL_FLAG=1
    fi
done

echo ""

if [ "$FAIL_FLAG" -eq 1 ]; then
    echo "============================================================"
    echo " ⚠️  部分进程失败！请检查以下日志："
    echo "============================================================"
    for ((gpu=0; gpu<GPU_COUNT; gpu++)); do
        if [ -f "${OUT_DIR}/gpu${gpu}.log" ]; then
            LAST_LINE=$(tail -1 "${OUT_DIR}/gpu${gpu}.log" 2>/dev/null || echo "(empty)")
            echo "  GPU ${gpu}: ${LAST_LINE}"
        fi
    done
    exit 1
fi

# ── 合并结果 ──────────────────────────────────────────────────────────────
MERGED="${OUT_DIR}/merged_n${TOTAL_N}_${TIMESTAMP}.jsonl"
echo "============================================================"
echo " 合并 ${GPU_COUNT} 个分片结果 → ${MERGED}"
echo "============================================================"

# 构建输入参数：收集所有 gpu*.jsonl 文件
INPUT_FILES=()
for ((gpu=0; gpu<GPU_COUNT; gpu++)); do
    f="${OUT_DIR}/gpu${gpu}_n*.jsonl"
    # shellcheck disable=SC2206
    INPUT_FILES+=($f)
done

python scripts/merge_results.py \
    "${INPUT_FILES[@]}" \
    --out "${MERGED}" \
    --no-dedupe \
    --renumber-ids

RECORDS=$(wc -l < "${MERGED}")
echo "  合并完成: ${RECORDS} 条记录"

# ── 生成分析摘要 ──────────────────────────────────────────────────────────
SUMMARY="${OUT_DIR}/summary_${TIMESTAMP}.json"
echo ""
echo "============================================================"
echo " 生成分析摘要 → ${SUMMARY}"
echo "============================================================"

python scripts/analyze_results.py \
    --in "${MERGED}" \
    --out "${SUMMARY}" 2>&1 || echo "  (analyze_results 跳过，可稍后手动运行)"

# ── 完成 ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " ✅ 全部完成！"
echo "============================================================"
echo "  合并结果 : ${MERGED}"
echo "  分析摘要 : ${SUMMARY}"
echo "  单 GPU 日志 : ${OUT_DIR}/gpu*.log"
echo "============================================================"
