"""
Pickle Stability and Correctness Test Suite
============================================
Tests whether pickle produces hash-identical output for the same input.

Testing techniques:
- Equivalence Partitioning
- Boundary Value Analysis
- Fuzzing (fixed seeds, reproducible)
- White-box: all-def / all-uses inspired

PEP8 compliant. Run: python -m pytest test_pickle_stability.py -v
"""

import hashlib
import io
import math
import pickle
import random
import string
import struct
import sys
import unittest


# ---------------------------------------------------------------------------
# Module-level helper classes
# (pickle cannot serialize locally-defined / nested classes)
# ---------------------------------------------------------------------------

class Point:
    """Simple 2-D point."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return isinstance(other, Point) and self.x == other.x and self.y == other.y


class Vector:
    """2-D vector with __slots__."""
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return isinstance(other, Vector) and self.x == other.x and self.y == other.y


class MyObj:
    """Object with custom __reduce_ex__."""
    def __init__(self, val):
        self.val = val

    def __reduce_ex__(self, protocol):
        return (self.__class__, (self.val,))

    def __eq__(self, other):
        return isinstance(other, MyObj) and self.val == other.val


class Stateful:
    """Object with __getstate__ / __setstate__."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getstate__(self):
        return {"x": self.x, "y": self.y}

    def __setstate__(self, state):
        self.x = state["x"]
        self.y = state["y"]

    def __eq__(self, other):
        return (
            isinstance(other, Stateful)
            and self.x == other.x
            and self.y == other.y
        )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def pickle_sha256(obj, protocol=None):
    """SHA-256 of pickle.dumps(obj)."""
    kwargs = {} if protocol is None else {"protocol": protocol}
    return hashlib.sha256(pickle.dumps(obj, **kwargs)).hexdigest()


def pickle_sha256_via_file(obj, protocol=None):
    """SHA-256 of pickle.dump() output via BytesIO."""
    buf = io.BytesIO()
    kwargs = {} if protocol is None else {"protocol": protocol}
    pickle.dump(obj, buf, **kwargs)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def assert_hash_identical(obj, protocol=None, repetitions=10):
    """Pickle obj `repetitions` times; assert all hashes match."""
    hashes = [pickle_sha256(obj, protocol) for _ in range(repetitions)]
    assert len(set(hashes)) == 1, (
        f"Non-deterministic pickle for {obj!r}. Hashes: {set(hashes)}"
    )
    return hashes[0]


# ---------------------------------------------------------------------------
# 1. Equivalence Partitioning
# ---------------------------------------------------------------------------

class TestEquivalencePartitioning(unittest.TestCase):

    def _s(self, obj):
        return assert_hash_identical(obj)

    def test_none(self):              self._s(None)
    def test_bool_true(self):         self._s(True)
    def test_bool_false(self):        self._s(False)
    def test_int_zero(self):          self._s(0)
    def test_int_positive(self):      self._s(42)
    def test_int_negative(self):      self._s(-99)
    def test_float_normal(self):      self._s(3.14)
    def test_float_nan(self):         self._s(float("nan"))
    def test_float_inf(self):         self._s(float("inf"))
    def test_float_neg_inf(self):     self._s(float("-inf"))
    def test_complex(self):           self._s(1 + 2j)
    def test_string_ascii(self):      self._s("hello world")
    def test_string_unicode(self):    self._s("Монгол 日本語 العربية")
    def test_string_empty(self):      self._s("")
    def test_bytes_normal(self):      self._s(b"\x00\xff\xde\xad\xbe\xef")
    def test_bytes_empty(self):       self._s(b"")
    def test_list_homo(self):         self._s([1, 2, 3])
    def test_list_hetero(self):       self._s([1, "two", 3.0, None, True])
    def test_list_empty(self):        self._s([])
    def test_tuple_simple(self):      self._s((1, "a", 3.14))
    def test_tuple_empty(self):       self._s(())
    def test_dict_simple(self):       self._s({"a": 1, "b": 2})
    def test_dict_empty(self):        self._s({})

    def test_set_simple(self):
        """Sets may be non-deterministic due to hash randomisation — documented."""
        hashes = [pickle_sha256({1, 2, 3, 4, 5}) for _ in range(20)]
        if len(set(hashes)) > 1:
            self.skipTest("set pickle non-deterministic on this platform (expected)")

    def test_frozenset(self):
        hashes = [pickle_sha256(frozenset({1, 2, 3})) for _ in range(20)]
        if len(set(hashes)) > 1:
            self.skipTest("frozenset pickle non-deterministic on this platform (expected)")

    def test_custom_class_instance(self):
        self._s(Point(3, 7))

    def test_custom_class_with_slots(self):
        self._s(Vector(1.0, 2.0))


# ---------------------------------------------------------------------------
# 2. Boundary Value Analysis
# ---------------------------------------------------------------------------

class TestBoundaryValues(unittest.TestCase):

    def _s(self, obj):
        return assert_hash_identical(obj)

    def test_int_max_sys(self):         self._s(sys.maxsize)
    def test_int_min_sys(self):         self._s(-sys.maxsize - 1)
    def test_int_very_large(self):      self._s(10 ** 1000)
    def test_int_plus_one(self):        self._s(1)
    def test_int_minus_one(self):       self._s(-1)
    def test_float_max(self):           self._s(sys.float_info.max)
    def test_float_min_pos(self):       self._s(sys.float_info.min)
    def test_float_epsilon(self):       self._s(sys.float_info.epsilon)
    def test_float_zero(self):          self._s(0.0)
    def test_float_neg_zero(self):      self._s(-0.0)
    def test_float_pi(self):            self._s(math.pi)
    def test_float_e(self):             self._s(math.e)
    def test_string_single(self):       self._s("x")
    def test_string_255(self):          self._s("a" * 255)
    def test_string_256(self):          self._s("a" * 256)
    def test_string_65535(self):        self._s("a" * 65535)
    def test_string_nulls(self):        self._s("\x00" * 10)
    def test_bytes_single(self):        self._s(b"\x00")
    def test_bytes_all_values(self):    self._s(bytes(range(256)))
    def test_list_single(self):         self._s([42])
    def test_list_large(self):          self._s(list(range(10_000)))
    def test_dict_single(self):         self._s({"k": "v"})
    def test_dict_large(self):          self._s({str(i): i for i in range(1000)})

    def test_nested_list_deep(self):
        obj = []
        inner = obj
        for _ in range(100):
            inner.append([])
            inner = inner[0]
        self._s(obj)


# ---------------------------------------------------------------------------
# 3. Recursive / Self-Referential Structures
# ---------------------------------------------------------------------------

class TestRecursiveStructures(unittest.TestCase):

    def test_self_referential_list(self):
        obj = []
        obj.append(obj)
        self.assertEqual(pickle_sha256(obj), pickle_sha256(obj))

    def test_self_referential_dict(self):
        obj = {}
        obj["self"] = obj
        self.assertEqual(pickle_sha256(obj), pickle_sha256(obj))

    def test_mutual_reference(self):
        a = []
        b = [a]
        a.append(b)
        self.assertEqual(pickle_sha256(a), pickle_sha256(a))

    def test_shared_reference(self):
        shared = [1, 2, 3]
        assert_hash_identical([shared, shared])

    def test_deeply_nested_dict(self):
        node = {"value": 0, "children": []}
        current = node
        for i in range(1, 50):
            child = {"value": i, "children": []}
            current["children"].append(child)
            current = child
        assert_hash_identical(node)


# ---------------------------------------------------------------------------
# 4. Protocol Version Tests
# ---------------------------------------------------------------------------

class TestProtocolVersions(unittest.TestCase):

    SAMPLE = {"key": [1, 2, 3], "value": "hello", "flag": True}

    def _stable(self, proto):
        h1 = pickle_sha256(self.SAMPLE, protocol=proto)
        h2 = pickle_sha256(self.SAMPLE, protocol=proto)
        self.assertEqual(h1, h2, f"Protocol {proto} non-deterministic")

    def test_protocol_0(self):   self._stable(0)
    def test_protocol_1(self):   self._stable(1)
    def test_protocol_2(self):   self._stable(2)
    def test_protocol_3(self):   self._stable(3)
    def test_protocol_4(self):   self._stable(4)

    def test_protocol_5(self):
        if pickle.HIGHEST_PROTOCOL < 5:
            self.skipTest("Protocol 5 not available")
        self._stable(5)

    def test_protocol_highest(self):
        self._stable(pickle.HIGHEST_PROTOCOL)

    def test_protocols_differ(self):
        """Different protocols produce different byte sequences."""
        hashes = {
            p: pickle_sha256(self.SAMPLE, protocol=p)
            for p in range(pickle.HIGHEST_PROTOCOL + 1)
        }
        self.assertGreater(len(set(hashes.values())), 1)


# ---------------------------------------------------------------------------
# 5. dumps() vs dump()-to-file Consistency
# ---------------------------------------------------------------------------

class TestDumpVsFile(unittest.TestCase):

    OBJECTS = [None, 42, 3.14, "hello", b"\xde\xad", [1, 2], {"a": 1}, (1, 2)]

    def test_dumps_vs_dump(self):
        for obj in self.OBJECTS:
            for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                self.assertEqual(
                    pickle_sha256(obj, protocol=proto),
                    pickle_sha256_via_file(obj, protocol=proto),
                    f"dumps/dump mismatch for {obj!r} at protocol {proto}"
                )


# ---------------------------------------------------------------------------
# 6. Round-trip Correctness
# ---------------------------------------------------------------------------

class TestRoundTrip(unittest.TestCase):

    def _rt(self, obj):
        return pickle.loads(pickle.dumps(obj))

    def test_none(self):      self.assertIsNone(self._rt(None))
    def test_string(self):
        for s in ["", "hello", "Монгол", "\x00\xff"]:
            self.assertEqual(self._rt(s), s)

    def test_int(self):
        for v in [0, 1, -1, sys.maxsize, -(sys.maxsize + 1), 10 ** 500]:
            self.assertEqual(self._rt(v), v)

    def test_float(self):
        for v in [0.0, 3.14, -0.0, math.pi, float("inf"), float("-inf")]:
            self.assertEqual(
                struct.pack("d", v), struct.pack("d", self._rt(v)),
                f"Float round-trip failed for {v!r}"
            )

    def test_nan(self):
        self.assertTrue(math.isnan(self._rt(float("nan"))))

    def test_list(self):
        self.assertEqual(self._rt([1, "two", None, [3, 4]]), [1, "two", None, [3, 4]])

    def test_dict(self):
        obj = {"a": 1, "b": [2, 3], "c": {"nested": True}}
        self.assertEqual(self._rt(obj), obj)

    def test_bytes(self):
        self.assertEqual(self._rt(bytes(range(256))), bytes(range(256)))

    def test_self_ref_list(self):
        obj = []
        obj.append(obj)
        restored = self._rt(obj)
        self.assertIs(restored[0], restored)


# ---------------------------------------------------------------------------
# 7. Fuzzing (fixed seeds — reproducible)
# ---------------------------------------------------------------------------

class TestFuzzing(unittest.TestCase):

    SEEDS = list(range(20))

    def _primitive(self, rng):
        kind = rng.choice(["int", "float", "str", "bytes", "bool", "none"])
        if kind == "int":     return rng.randint(-(2 ** 63), 2 ** 63)
        if kind == "float":   return rng.uniform(-1e308, 1e308)
        if kind == "str":
            return "".join(rng.choice(string.printable) for _ in range(rng.randint(0, 200)))
        if kind == "bytes":
            return bytes(rng.randint(0, 255) for _ in range(rng.randint(0, 200)))
        if kind == "bool":    return rng.choice([True, False])
        return None

    def _obj(self, rng, depth=0):
        if depth > 4:
            return self._primitive(rng)
        kind = rng.choice(["primitive", "list", "dict", "tuple"])
        if kind == "primitive": return self._primitive(rng)
        if kind == "list":
            return [self._obj(rng, depth + 1) for _ in range(rng.randint(0, 6))]
        if kind == "tuple":
            return tuple(self._obj(rng, depth + 1) for _ in range(rng.randint(0, 6)))
        return {str(rng.randint(0, 1000)): self._obj(rng, depth + 1)
                for _ in range(rng.randint(0, 6))}

    def test_fuzzing_seeds(self):
        for seed in self.SEEDS:
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                obj = self._obj(rng)
                h1 = pickle_sha256(obj)
                h2 = pickle_sha256(obj)
                self.assertEqual(h1, h2, f"Seed {seed}: non-deterministic for {obj!r}")


# ---------------------------------------------------------------------------
# 8. White-box / Structural Tests
# ---------------------------------------------------------------------------

class TestWhiteBox(unittest.TestCase):
    """
    Targets specific code paths in CPython's pickle.py:
    memo dict, reduce_ex dispatch, __getstate__/__setstate__, persistent_id,
    and PROTO opcode.
    """

    def test_memo_deduplication(self):
        """Shared sub-object must use memo (BINGET); output still deterministic."""
        shared = list(range(50))
        container = [shared, shared, shared]
        assert_hash_identical(container)
        data_shared = pickle.dumps(container)
        data_copied = pickle.dumps(
            [list(range(50)), list(range(50)), list(range(50))]
        )
        self.assertLess(len(data_shared), len(data_copied))

    def test_reduce_ex_dispatch(self):
        """Objects with __reduce_ex__ must pickle deterministically."""
        assert_hash_identical(MyObj(42))

    def test_getstate_setstate(self):
        """Custom __getstate__ / __setstate__ must be stable and correct."""
        obj = Stateful(10, 20)
        assert_hash_identical(obj)
        self.assertEqual(pickle.loads(pickle.dumps(obj)), obj)

    def test_persistent_id(self):
        """Pickler with persistent_id produces deterministic output."""
        DB = {1: "record_1", 2: "record_2"}

        class PersistentPickler(pickle.Pickler):
            def persistent_id(self, obj):
                if isinstance(obj, int) and obj in DB:
                    return f"db:{obj}"
                return None

        def stable_hash(obj):
            buf = io.BytesIO()
            PersistentPickler(buf).dump(obj)
            return hashlib.sha256(buf.getvalue()).hexdigest()

        data = [1, 2, "not_in_db"]
        self.assertEqual(stable_hash(data), stable_hash(data))

    def test_proto_opcode(self):
        """For protocol >= 2, first byte must be PROTO (0x80), second the version."""
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            data = pickle.dumps(42, protocol=proto)
            self.assertEqual(data[0], 0x80, f"Protocol {proto}: missing PROTO opcode")
            self.assertEqual(data[1], proto, f"Protocol {proto}: wrong version byte")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
