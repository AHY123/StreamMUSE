import math


SECONDS_PER_TICK = (60.0 / 120.0) / 4.0
EARLY_SLEEP_SECONDS = SECONDS_PER_TICK * 0.1


def assign_live_ticks(event_times, total_ticks):
    assigned = []
    pending = list(sorted(event_times))
    queue = []
    next_event = 0

    for tick_count in range(total_ticks):
        process_time = (tick_count * SECONDS_PER_TICK) + EARLY_SLEEP_SECONDS

        while next_event < len(pending) and pending[next_event] <= process_time:
            queue.append(pending[next_event])
            next_event += 1

        while queue:
            event_time = queue.pop(0)
            assigned.append(
                {
                    "event_time_s": round(event_time, 4),
                    "assigned_tick": tick_count,
                    "assigned_tick_time_s": round(tick_count * SECONDS_PER_TICK, 4),
                    "tick_error": tick_count - math.floor(event_time / SECONDS_PER_TICK),
                }
            )

    return assigned


def assign_midi_ticks(explicit_ticks):
    return [
        {
            "scheduled_tick": tick,
            "assigned_tick": tick,
            "tick_error": 0,
        }
        for tick in explicit_ticks
    ]


def sweep_one_tick(sample_count=100):
    counts = {0: 0, 1: 0}
    for i in range(sample_count):
        event_time = (i / sample_count) * SECONDS_PER_TICK
        assigned_tick = assign_live_ticks([event_time], total_ticks=2)[0]["assigned_tick"]
        counts[assigned_tick] += 1
    return counts


def main():
    print("Lekai client timing analysis")
    print(f"seconds_per_tick={SECONDS_PER_TICK:.4f}")
    print(f"early_sleep_seconds={EARLY_SLEEP_SECONDS:.4f}")
    print()

    live_rhythm = [0.0500, 0.1750, 0.3000, 0.4250]
    midi_ticks = [0, 1, 2, 3]

    print("Live input rhythm (same musical intent, no explicit tick):")
    for rec in assign_live_ticks(live_rhythm, total_ticks=5):
        print(rec)
    print()

    print("MIDI-file rhythm (same musical intent, explicit tick):")
    for rec in assign_midi_ticks(midi_ticks):
        print(rec)
    print()

    sweep = sweep_one_tick(sample_count=100)
    print("Uniform sweep across one tick:")
    print(
        {
            "assigned_same_tick_samples": sweep[0],
            "assigned_next_tick_samples": sweep[1],
        }
    )


if __name__ == "__main__":
    main()
