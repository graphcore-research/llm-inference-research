# Part 1 (main)

Command:

```
python run_sweep.py --model MODEL --task TASK
```

Run status:

```
|                | Llama-2-7b-hf | pythia-6.9b | pythia-2.8b | pythia-1.4b |
| repetition     |          DONE |        DONE |        DONE |        DONE |
| squad          |          DONE |        DONE |        DONE |        DONE |
| triviaqa       |          DONE |        DONE |        DONE |        DONE |
| wikitext_bpc   |          DONE |        DONE |        DONE |        DONE |
| cnn_dailymail  |          DONE |        DONE |        DONE |        DONE |
```

---

# Part 2 (local)

Command:

```
python run_sweep_v2.py -k K0 K1
```

Run status:

```
|    k |  status | rerun (llama-{triviaqa, squad}) |
|  128 |    DONE |                            DONE |
|  192 |    DONE |                            DONE |
|  256 |    DONE |                            DONE |
|  384 |    DONE |                            DONE |
|  512 |    DONE |                            DONE |
|  768 |    DONE |                            DONE |
```
