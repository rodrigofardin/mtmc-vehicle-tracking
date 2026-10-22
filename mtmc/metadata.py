"""Leitura tolerante dos arquivos .txt de metadados (cam_timestamp, cam_framenum).

Formato verificado no dataset real (S01): uma linha por câmera, `<cam_id> <valor>`,
separado por espaço, terminador CRLF, ASCII, sem comentários. O parser também
aceita tab/vírgula/`;`, BOM, linhas vazias e comentários iniciados por `#`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

_SEP = re.compile(r"[\s,;]+")


class MetadataError(ValueError):
    """Metadado ausente, ilegível ou inválido."""


@dataclass
class ParsedTable:
    values: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    source: str = ""


def parse_table(path, *, integer: bool = False, allow_negative: bool = True) -> ParsedTable:
    p = Path(path)
    if not p.is_file():
        raise MetadataError(f"arquivo de metadados não encontrado: {p}")
    try:
        text = p.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise MetadataError(f"encoding inválido em {p}: {e}") from e
    values: Dict[str, float] = {}
    warnings: List[str] = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [x for x in _SEP.split(line) if x]
        if len(parts) != 2:
            raise MetadataError(f"{p}:{n}: esperado 2 colunas, obtido {len(parts)}: {raw!r}")
        cam, val = parts
        try:
            num = float(val)
        except ValueError as e:
            raise MetadataError(f"{p}:{n}: valor não numérico {val!r}") from e
        if num != num or num in (float("inf"), float("-inf")):
            raise MetadataError(f"{p}:{n}: valor não finito {val!r}")
        if integer and num != int(num):
            raise MetadataError(f"{p}:{n}: esperado inteiro, obtido {val!r}")
        if not allow_negative and num < 0:
            raise MetadataError(f"{p}:{n}: valor negativo {val!r}")
        if cam in values:
            warnings.append(f"{p}:{n}: câmera {cam} duplicada; mantido o último valor")
        values[cam] = int(num) if integer else num
    if not values:
        raise MetadataError(f"nenhuma entrada válida em {p}")
    return ParsedTable(values, warnings, str(p))


def read_timestamps(path) -> ParsedTable:
    return parse_table(path)


def read_framenums(path) -> ParsedTable:
    return parse_table(path, integer=True, allow_negative=False)
