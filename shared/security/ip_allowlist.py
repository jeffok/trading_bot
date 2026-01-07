from __future__ import annotations

import ipaddress
from typing import Iterable, Tuple


def parse_allowlist(items: Iterable[str]) -> Tuple[ipaddress._BaseNetwork, ...]:
    nets = []
    for raw in items:
        s = str(raw).strip()
        if not s:
            continue
        try:
            # if plain IP, treat as /32 or /128
            if "/" not in s:
                ip = ipaddress.ip_address(s)
                s = f"{s}/32" if ip.version == 4 else f"{s}/128"
            net = ipaddress.ip_network(s, strict=False)
            nets.append(net)
        except Exception:
            # ignore invalid entries silently (configuration error should not crash)
            continue
    return tuple(nets)


def is_ip_allowed(ip: str, allowlist: Iterable[str]) -> bool:
    items = list(allowlist)
    if not items:
        return True
    try:
        addr = ipaddress.ip_address(ip)
    except Exception:
        return False
    nets = parse_allowlist(items)
    if not nets:
        # if allowlist configured but all invalid -> deny by default
        return False
    return any(addr in n for n in nets)
