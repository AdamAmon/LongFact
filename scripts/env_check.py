import importlib

def safe_import(name):
    try:
        mod = importlib.import_module(name)
        return mod, None
    except Exception as e:
        return None, str(e)


def main():
    mods = ['torch', 'transformers', 'sentence_transformers']
    info = {}
    for m in mods:
        mod, err = safe_import(m)
        if mod:
            if m == 'torch':
                info['torch_version'] = getattr(mod, '__version__', 'unknown')
                try:
                    info['cuda_available'] = mod.cuda.is_available()
                except Exception:
                    info['cuda_available'] = False
            else:
                info[f'{m}_version'] = getattr(mod, '__version__', 'unknown')
        else:
            info[f'{m}_import_error'] = err

    import json
    print(json.dumps(info, ensure_ascii=False))


if __name__ == '__main__':
    main()
