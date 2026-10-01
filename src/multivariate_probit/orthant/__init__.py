"""orthant -- multivariate-normal orthant probabilities, P(X <= upper).

On import this binds ``orthant.cdf`` to one of two binaries:

* a valid key present  -> the paid binary is decrypted in memory and bound;
  full accuracy, unlimited dimension.
* no key / bad key     -> the plaintext free binary is bound (n <= 3,
  resolution="low" only) and a one-time notice is printed to stderr.

The key is read from ``$ORTHANT_KEY`` or ``~/.orthant/key``.  Get one at
https://quantecarlo.com/orthant_key
"""
import glob
import importlib.util
import os
import struct
import sys
import tempfile

__all__ = ["cdf"]

_URL = "https://quantecarlo.com/orthant_key"
_DIR = os.path.dirname(os.path.abspath(__file__))
_MAGIC = b"ORT1"
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2 ** 14, 8, 1

tier = "free"  # set to "paid" when the paid binary is bound


def _read_key():
    k = os.environ.get("ORTHANT_KEY")
    if k and k.strip():
        return k.strip()
    try:
        with open(os.path.expanduser("~/.orthant/key")) as f:
            k = f.read().strip()
            return k or None
    except OSError:
        return None


def _import_extension(data, modname):
    """Import an extension module from raw bytes without leaving it on disk.

    Uses an anonymous in-memory file (memfd) on Linux so the decrypted binary
    never touches the filesystem; falls back to a short-lived temp file on
    platforms without memfd.
    """
    qualified = __name__ + "." + modname
    if hasattr(os, "memfd_create"):
        fd = os.memfd_create(modname)
        try:
            os.write(fd, data)
            spec = importlib.util.spec_from_file_location(
                qualified, "/proc/self/fd/%d" % fd)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        finally:
            os.close(fd)
    tmp = tempfile.NamedTemporaryFile(suffix=".so", delete=False)
    try:
        tmp.write(data)
        tmp.close()
        spec = importlib.util.spec_from_file_location(qualified, tmp.name)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


def _decrypt_paid(passphrase):
    """Return the decrypted paid binary bytes, or raise on a bad key/blob."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

    with open(os.path.join(_DIR, "_paid.enc"), "rb") as f:
        blob = f.read()
    if blob[:4] != _MAGIC:
        raise ValueError("paid blob: bad magic")
    (version,) = struct.unpack("B", blob[4:5])
    if version != 1:
        raise ValueError("paid blob: unsupported version %d" % version)
    salt, nonce, ct = blob[5:21], blob[21:33], blob[33:]
    key = Scrypt(salt=salt, length=32, n=_SCRYPT_N, r=_SCRYPT_R,
                 p=_SCRYPT_P).derive(passphrase.encode("utf-8"))
    return AESGCM(key).decrypt(nonce, ct, None)  # raises InvalidTag on wrong key


def _load_free():
    matches = glob.glob(os.path.join(_DIR, "_ofree.*.so"))
    if not matches:
        raise ImportError("orthant: free binary missing from package")
    spec = importlib.util.spec_from_file_location(
        __name__ + "._ofree", matches[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _warn(msg):
    try:
        sys.stderr.write(msg)
        sys.stderr.flush()
    except Exception:
        pass


def _bind():
    """Pick a binary, bind cdf, and return the tier."""
    key = _read_key()
    if key is not None:
        try:
            mod = _import_extension(_decrypt_paid(key), "_ocore")
        except ImportError as e:
            # Not a key problem: a missing dependency or a binary that won't
            # load on this platform.  Say so rather than blaming the key.
            notice = ("orthant: key found but the paid binary could not be "
                      "loaded (%s) -- running in FREE mode (n<=3, "
                      "resolution='low').\n" % e)
            if (e.name or "").startswith("cryptography"):
                notice += "         pip install cryptography\n"
        except Exception:
            notice = ("orthant: key invalid or unreadable -- running in FREE "
                      "mode (n<=3, resolution='low').\n"
                      "         Renew or get a key: %s\n" % _URL)
        else:
            globals()["cdf"] = mod.cdf
            return "paid"
    else:
        notice = ("orthant: FREE mode -- n<=3 and resolution='low' only.\n"
                  "         For full accuracy and higher dimensions, get a "
                  "key: %s\n" % _URL)

    globals()["cdf"] = _load_free().cdf
    _warn(notice)
    return "free"


tier = _bind()
