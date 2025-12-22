def intersect(a, b):
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    return (start, end) if start < end else None

hold_music_intervals = []

for ns in non_speech_intervals:
    for non_silent in nonsilent_ranges:
        inter = intersect(ns, non_silent)
        if inter:
            hold_music_intervals.append(inter)

MERGE_GAP_MS = 300  # allow tiny gaps

merged = []

for start, end in sorted(hold_music_intervals):
    if not merged:
        merged.append([start, end])
    elif start - merged[-1][1] <= MERGE_GAP_MS:
        merged[-1][1] = max(merged[-1][1], end)
    else:
        merged.append([start, end])

merged = [tuple(x) for x in merged]


if merged:
    hold_start, hold_end = max(merged, key=lambda x: x[1] - x[0])
    hold_duration_sec = (hold_end - hold_start) / 1000
else:
    hold_start = hold_end = hold_duration_sec = 0


print("Hold music start (ms):", hold_start)
print("Hold music end (ms):", hold_end)
print("Hold music duration (sec):", hold_duration_sec)
