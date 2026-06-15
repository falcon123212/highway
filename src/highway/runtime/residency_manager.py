from collections import OrderedDict
from typing import Dict, Optional


class ResidencyManager:
    def __init__(self, max_resident_bytes: Optional[int] = None):
        self.max_resident_bytes = max_resident_bytes
        self._resident: "OrderedDict[str, int]" = OrderedDict()
        self._resident_bytes = 0
        self.bytes_read = 0
        self.blocks_materialized = 0
        self.hotset_hits = 0
        self.evictions = 0
        self.max_resident_observed_bytes = 0

    def admit(self, key: str, size_bytes: int) -> bool:
        size = max(0, int(size_bytes))
        if key in self._resident:
            self.hotset_hits += 1
            self._resident.move_to_end(key)
            return True

        self._resident[key] = size
        self._resident_bytes += size
        self.bytes_read += size
        self.blocks_materialized += 1
        self.max_resident_observed_bytes = max(self.max_resident_observed_bytes, self._resident_bytes)

        if self.max_resident_bytes is not None:
            while self._resident_bytes > self.max_resident_bytes and self._resident:
                _, evicted_size = self._resident.popitem(last=False)
                self._resident_bytes -= evicted_size
                self.evictions += 1
        return False

    def snapshot_metrics(self) -> Dict[str, float]:
        return {
            "bytes_read": self.bytes_read,
            "blocks_materialized": self.blocks_materialized,
            "hotset_hits": self.hotset_hits,
            "evictions": self.evictions,
            "resident_bytes": self._resident_bytes,
            "max_resident_mb": round(self.max_resident_observed_bytes / (1024 * 1024), 6),
        }
