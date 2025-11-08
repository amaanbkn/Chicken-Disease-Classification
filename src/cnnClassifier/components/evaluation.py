# inside Evaluation class
import os, glob, tensorflow as tf # type: ignore

def _resolve_model_path(self, path_like: str, ckpt_dir: str | None = None) -> str:
    p = str(path_like).replace("\\", "/")
    root, ext = os.path.splitext(p)

    if os.path.isdir(p):
        cands = sorted(
            glob.glob(os.path.join(p, "**", "*.keras"), recursive=True) +
            glob.glob(os.path.join(p, "**", "*.h5"), recursive=True),
            key=os.path.getmtime
        )
        if cands:
            return cands[-1]
        raise FileNotFoundError(f"No .keras/.h5 in dir: {p}")

    if ext == "":
        for e in (".keras", ".h5"):
            cand = root + e
            if os.path.isfile(cand):
                return cand

    if not os.path.isfile(p):
        alt = root + (".keras" if ext.lower() != ".keras" else ".h5")
        if os.path.isfile(alt):
            return alt

    if ckpt_dir:
        ckpt_dir = str(ckpt_dir).replace("\\", "/")
        ckpts = sorted(
            glob.glob(os.path.join(ckpt_dir, "**", "*.keras"), recursive=True) +
            glob.glob(os.path.join(ckpt_dir, "**", "*.h5"), recursive=True),
            key=os.path.getmtime
        )
        if ckpts:
            return ckpts[-1]

    if os.path.isfile(p):
        return p
    raise FileNotFoundError(f"Model not found. Tried: {p} / {root+'.keras'} / {root+'.h5'}")

def load_model(self, path: str, ckpt_dir: str | None = None):
    resolved = self._resolve_model_path(path, ckpt_dir)
    print(f"✅ Loading model from: {resolved}")
    return tf.keras.models.load_model(resolved)
model = Self.load_model(self.config.path_of_model, # type: ignore
                        getattr(self.config, "checkpoint_dir", None)) # type: ignore
