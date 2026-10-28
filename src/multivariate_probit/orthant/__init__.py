"""orthant -- multivariate-normal orthant probabilities, P(X <= upper).

On import this binds ``orthant.cdf`` to one of two binaries:

* a valid key present  -> the paid binary is decrypted in memory and bound;
  full accuracy, unlimited dimension.
* no key / bad key     -> the plaintext free binary is bound (n <= 3,
  resolution="low" only) and a one-time notice is printed to stderr on the
  first call.

The key is read from ``$ORTHANT_KEY`` or ``~/.orthant/key`` at import, or
passed at any time with ``activate(key)`` (no environment needed, no restart):

    >>> from multivariate_probit import orthant
    >>> orthant.activate("my-key")
    'paid'

Get a key at https://quantecarlo.com/orthant_key
"""
import glob
import importlib.machinery
import importlib.util
import os
import struct
import sys
import tempfile

__all__ = ["cdf", "activate"]

_URL = "https://quantecarlo.com/orthant_key"
_DIR = os.path.dirname(os.path.abspath(__file__))
_MAGIC = b"ORT1"
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2 ** 14, 8, 1

tier = "free"  # set to "paid" when the paid binary is bound
_notice = None  # why we are in free mode; printed once, on the first free call


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
            path = "/proc/self/fd/%d" % fd
            # The path has no .so suffix, so the loader must be given
            # explicitly; inferred from the path, the spec would be None.
            spec = importlib.util.spec_from_file_location(
                qualified, path,
                loader=importlib.machinery.ExtensionFileLoader(qualified, path))
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


def _tag():
    """Extension tag of the first shipped binary this interpreter can load,
    e.g. "cpython-311-x86_64-linux-gnu"."""
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        if suffix.count(".") == 2 and suffix.endswith(".so"):
            tag = suffix[1:-3]
            if os.path.exists(os.path.join(_DIR, "_ofree.%s.so" % tag)):
                return tag
    shipped = sorted(os.path.basename(p)[len("_ofree."):-len(".so")]
                     for p in glob.glob(os.path.join(_DIR, "_ofree.*.so")))
    raise ImportError(
        "orthant: no binary for Python %d.%d on this platform; shipped: %s"
        % (sys.version_info[0], sys.version_info[1],
           ", ".join(shipped) or "none"))


def _decrypt_paid(passphrase, tag):
    """Return the decrypted paid binary bytes, or raise on a bad key/blob."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

    with open(os.path.join(_DIR, "_paid.%s.enc" % tag), "rb") as f:
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


def _load_free(tag):
    spec = importlib.util.spec_from_file_location(
        __name__ + "._ofree", os.path.join(_DIR, "_ofree.%s.so" % tag))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _warn(msg):
    try:
        sys.stderr.write(msg)
        sys.stderr.flush()
    except Exception:
        pass


def _free_cdf(free):
    """The free binary's cdf, printing the free-mode notice on its first call."""
    def cdf(*args, **kwargs):
        global _notice
        if _notice:
            _warn(_notice)
            _notice = None
        return free(*args, **kwargs)
    cdf.__doc__ = free.__doc__
    return cdf


def activate(key):
    """Bind the paid binary with ``key``, at any time; return ``"paid"``.

    Unlike the import-time lookup this never falls back: a wrong key raises
    ``ValueError`` and a binary that cannot load here raises ``ImportError``,
    leaving the current binding unchanged.
    """
    global tier, _notice
    if not isinstance(key, str) or not key.strip():
        raise ValueError("orthant: activate() needs a non-empty key string")
    try:
        data = _decrypt_paid(key.strip(), _tag())
    except ImportError as e:
        if (e.name or "").startswith("cryptography"):
            raise ImportError("orthant: activate() needs the cryptography package: "
                              "pip install cryptography (or multivariate-probit[orthant])",
                              name=e.name) from None
        raise
    except Exception as e:
        raise ValueError("orthant: key not accepted by this binary (%s). "
                         "Renew or get a key: %s" % (type(e).__name__, _URL)) from None
    globals()["cdf"] = _import_extension(data, "_ocore").cdf
    tier, _notice = "paid", None
    return tier


def _bind():
    """Pick a binary at import, bind cdf, and return the tier."""
    global _notice
    tag = _tag()
    key = _read_key()
    if key is not None:
        try:
            data = _decrypt_paid(key, tag)
        except ImportError as e:
            notice = ("orthant: key found but the paid binary could not be "
                      "decrypted (%s) -- running in FREE mode (n<=3, "
                      "resolution='low').\n" % e)
            if (e.name or "").startswith("cryptography"):
                notice += "         pip install cryptography\n"
        except Exception:
            notice = ("orthant: key invalid or unreadable -- running in FREE "
                      "mode (n<=3, resolution='low').\n"
                      "         Renew or get a key: %s\n" % _URL)
        else:
            try:
                globals()["cdf"] = _import_extension(data, "_ocore").cdf
                return "paid"
            except Exception as e:
                # The key was right; loading the binary failed. Say so
                # rather than blaming the key.
                notice = ("orthant: key accepted but the paid binary could not "
                          "be loaded (%s: %s) -- running in FREE mode (n<=3, "
                          "resolution='low').\n" % (type(e).__name__, e))
    else:
        notice = ("orthant: FREE mode -- n<=3 and resolution='low' only. No key "
                  "in $ORTHANT_KEY or ~/.orthant/key when this module was "
                  "imported;\n"
                  "         pass one with orthant.activate(key). Get a key: "
                  "%s\n" % _URL)

    globals()["cdf"] = _free_cdf(_load_free(tag).cdf)
    _notice = notice
    return "free"


tier = _bind()
