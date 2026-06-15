# Cleanup Notes

## Archived Windows Path Shadow

The workspace previously contained a parasite directory named like a Windows drive path:

```text
c_drive_shadow/Users/nicol/Documents/Highway/corpus_poc2_stress/
```

It has been archived without deletion under:

```text
artifacts/legacy/c_drive_shadow/
```

Inventory before archive:

- File count: 17
- Approximate size: 135264 bytes
- Contents: `documents/`, `index/`, and `questions/` for `corpus_poc2_stress`

A later cleanup can compare it against any canonical stress corpus before removing it.
