> Smoke run, 2026-11-24: 6 MAESTRO v3 test pieces, first 60 s each. Not the final benchmark (that uses all 177 test pieces at full length).

# MAESTRO-Wild benchmark

6 pieces, 6.0 min of audio per condition.

## Note F1 (onset)

| condition     |   oracle |   reference |
|:--------------|---------:|------------:|
| clean         |      100 |        97.9 |
| reverb_medium |      100 |        93.6 |
| reverb_long   |      100 |        92.4 |
| noise_20      |      100 |        96.2 |
| noise_10      |      100 |        91.5 |
| noise_5       |      100 |        83.6 |
| phone         |      100 |        96.1 |
| combined      |      100 |        77.1 |

## Note F1 with offset

| condition     |   oracle |   reference |
|:--------------|---------:|------------:|
| clean         |      100 |        86.6 |
| reverb_medium |      100 |        64   |
| reverb_long   |      100 |        72.4 |
| noise_20      |      100 |        83.6 |
| noise_10      |      100 |        72.1 |
| noise_5       |      100 |        65.2 |
| phone         |      100 |        82.1 |
| combined      |      100 |        38.9 |

## Speed (real-time factor, lower is faster)

| model     |   RTF |
|:----------|------:|
| reference | 0.335 |
