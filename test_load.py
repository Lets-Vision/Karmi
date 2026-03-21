import traceback, os, sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

try:
    import tf_keras as keras
    print(f"tf_keras version: {keras.__version__}")
except ImportError:
    import keras
    print(f"keras version: {keras.__version__}")

script_dir = os.path.dirname(os.path.realpath(__file__))
fname = os.path.join(script_dir, 'karmi64.h5')
print(f"\nCargando: {fname} ({os.path.getsize(fname)} bytes)")

try:
    m = keras.models.load_model(fname, compile=False)
    print(f"OK! Input: {m.input_shape}, Output: {m.output_shape}, Layers: {len(m.layers)}")
except Exception as e:
    print(f"FALLO 1: {type(e).__name__}: {e}")
    print("\nIntentando con custom_objects vacio...")
    try:
        m = keras.models.load_model(fname, compile=False, custom_objects={})
        print(f"OK (2)! Input: {m.input_shape}")
    except Exception as e2:
        print(f"FALLO 2: {type(e2).__name__}: {e2}")
        print("\nIntentando cargar solo pesos con h5py...")
        try:
            import h5py
            with h5py.File(fname, 'r') as f:
                print("Keys en h5:", list(f.keys()))
                if 'model_weights' in f:
                    print("model_weights layers:", list(f['model_weights'].keys())[:10])
                if 'model_config' in f.attrs:
                    config = f.attrs['model_config']
                    if isinstance(config, bytes):
                        config = config.decode('utf-8')
                    print("model_config (primeros 500 chars):")
                    print(config[:500])
        except Exception as e3:
            print(f"FALLO h5py: {e3}")
