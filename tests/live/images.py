import struct
import zlib

from ai_lab import Image


def red_square(size: int = 64) -> Image:
    """A solid red PNG built from the format, so no binary fixture is committed."""
    row = b'\x00' + b'\xff\x00\x00' * size
    raw = zlib.compress(row * size)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))

    header = struct.pack('>IIBBBBB', size, size, 8, 2, 0, 0, 0)
    return Image(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header) + chunk(b'IDAT', raw) + chunk(b'IEND', b''))
