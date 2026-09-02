# AGENTS.md — generare video con questa repo

Nota operativa per un agente che arriva qui a freddo. Il `README.md` documenta
la CLI per intero: questo file copre solo lo **stato di questa macchina** e i
**numeri misurati**, che nel README non ci sono.

La specifica della feature in corso (LoRA a runtime) sta in **`docs/WHAT.md`**.

## Stato dell'installazione

| | |
|---|---|
| Binario | `./h3` (già compilato; `make -j8` per rifarlo) |
| Modello | `./MiniMax-H3` → symlink alla snapshot nella cache HF |
| Pipeline installata | **solo FL2VA** (144 GB) |
| Pipeline mancante | **Ref2VA** (altri 144 GB) |
| Hardware | Apple M4 Pro, 48 GiB unified, max Metal buffer 27 GiB, no Metal 4 |
| FFmpeg/FFprobe | presenti in `/opt/homebrew/bin` |

`./MiniMax-H3` è un symlink dentro la cache HuggingFace, **non copiarlo**: la
repo HF intera è 498 GB (tre pipeline BF16 duplicate, nessuna quantizzazione
esiste). Il solo `FL2VA/transformer` è **62 GB** — è la ragione per cui
`--ssd-streaming` esiste. Per riscaricare:

```sh
hf download MiniMaxAI/MiniMax-H3 --include "FL2VA/*"
```

Non usare `--local-dir`: le versioni recenti di `hf` non fanno symlink dalla
cache e ti riscaricano tutto.

### Cosa funziona e cosa no

Funzionano `-p` (prompt→video+audio), `--first-frame`, `--last-frame`.

**Non funzionano** `--ref-image`, `--ref-video`, `--ref-audio` e affini: sono
il percorso Ref2VA, che richiede un DiT separato non installato. `h3.c:434`
lo tratta come opzionale, quindi non fallisce all'avvio — fallisce solo quando
lo chiedi. Verifica con `./h3 --info -d ./MiniMax-H3`: se `Ref2VA DiT` mostra
`0 files`, quelle opzioni sono fuori gioco.

## Vincoli da rispettare

- **Canvas**: larghezza e altezza multipli di 32, prodotto ≤ `768*1344`.
- **Frame**: allineati verso l'alto a `5 + 17n` → 5, 22, 39, 56, 73, 90. A 24 fps
  sono 0,21 / 0,92 / 1,63 / 2,33 / 3,04 / 3,75 secondi.
- **Aspect ratio**: `--first-frame` viene **stirata** sul canvas (`README:667`).
  Scala l'immagine allo stesso rapporto del canvas o esce deformata. Es.: una
  sorgente 7:9 va su `448x576`, non su un quadrato.
- `--reuse` e `--core-reuse` sono mutuamente esclusivi.
- `--ssd-streaming` non si combina con `--use-int8-row-fc2`.
- Le opzioni `--use-*` marcate M5 non fanno nulla qui (`Metal 4: no`).

## I percorsi "fusi" M5/NAX sono inerti su questa macchina

`gpu.tensorOpsEnabled` richiede che il nome del device Metal contenga la stringa
letterale `"M5"` (`h3_gpu.m:364-373`). Qui è un M4 Pro: il flag è sempre falso.

Conseguenza per chi legge il codice: ogni kernel "fuso" NAX **non è il percorso
che gira davvero**. Es. `h3_gpu_grouped_qkv_linear_rope_bf16` cade sempre sul
fallback `h3_gpu_linear_bf16` + `h3_gpu_grouped_qkv_rope_bf16`. Prima di
modificare o profilare un fast path fuso, verifica che sia raggiungibile.

## Tieni sempre `--ssd-streaming`

Su questa macchina è **gratis**. Misurato sul run da 56 frame:

```
BF16 SSD stream 574.937 GiB read in 134.342s (4.280 GiB/s), unhidden wait 0.006s
```

6 ms di attesa disco non nascosta su 739 s di denoise. La GPU è abbastanza
lenta rispetto all'SSD che il prefetch copre tutto. Senza streaming il DiT
vuole ~37 GiB residenti su 48 totali con working set raccomandato di 36: va
in swap. Non c'è motivo di spegnerlo.

Picchi di memoria reali: DiT 2,58 GiB, **video VAE decoder 9,37 GiB** ← il più
alto della pipeline, è lui il limite se sali di risoluzione, non i pesi.

## Modello di costo (misurato, M4 Pro)

Due punti reali a `448x576`:

| frame | steps / reuse / layers | valutazioni | denoise | totale |
|---:|---|---:|---:|---:|
| 22 | 20 / 2 / 45 | 11 | 199 s | **4 min 05 s** |
| 56 | 30 / 2 / 50 | 16 | 739 s | **13 min 23 s** |

Come stimare un run nuovo:

1. **Valutazioni del denoiser** — è il moltiplicatore che conta. A 20 step:
   `--reuse 1` → 20, `--reuse 2` → 11, `--reuse 3` → 8. Sale con gli step
   (30 step + reuse 2 = 16).
2. **Costo per valutazione** ≈ `0,82 s × frame × (layers/50)` a `448x576`.
   Scala circa linearmente coi frame; l'attention è quadratica sui token,
   quindi **considerala una stima al ribasso, ±15%**.
3. **Overhead fisso** ≈ 25 s (encoder + DiT load) **più il video VAE decoder**,
   che scala coi frame: 19 s a 22 frame, 45 s a 56.

Esempio: 39 frame, layers 50, 20 valutazioni → `0,82 × 39 × 1 = 32 s/val`
→ `32 × 20 = 640 s` + ~60 s = **~11,7 min**.

La risoluzione alza solo il tempo GPU, **non** il costo di streaming: i pesi
letti per valutazione sono gli stessi. Su questa macchina l'unica leva che
taglia davvero il tempo è ridurre le valutazioni.

## Ordine di priorità sulla qualità

Dal `README:162`, una modifica alla volta: prima `--layers 50`, poi tutte le
valutazioni (`--reuse 1`), e solo infine più step. Sotto budget stretto usa
`--steps 4..7` con `--reuse 1`: è una schedule dedicata ai budget bassi, meglio
di una da 20 step con riuso aggressivo.

## Prompt

La guida ufficiale è <https://fal.ai/learn/devs/minimax-h3-prompting-guide>.
In sintesi: elenca i dettagli che definiscono il soggetto, descrivi l'azione
come **evento fisico** e non come emozione astratta, usa lessico da
cinematografia per camera e luce, **dirigi l'audio con la stessa precisione del
visivo**, e chiudi con i vincoli. Blocchi timecoded (`[0-2 seconds] ...`) solo
oltre i 2-3 secondi. Limite 7.000 caratteri.

Su sorgenti illustrate serve un guard-rail esplicito in coda, del tipo
`keep the flat painterly illustration style with visible ink linework; do not
drift toward photorealism`: senza, H3 vira fotorealistico.

Esempi funzionanti in `prompts/*.txt`.

## Ricetta

```sh
./h3 --profile -d ./MiniMax-H3 \
  --first-frame refs/immagine.png \
  -p "$(cat prompts/01-breath.txt)" \
  --width 448 --height 576 \
  --frames 56 --steps 30 \
  --layers 50 --reuse 2 --ssd-streaming \
  -o outputs/nome.mp4 > logs/run.log 2>&1
```

Redirigi **sempre** su file. `h3` scrive avanzamento con `\r` e una pipe verso
`tail` bufferizza tutto fino all'uscita: il run diventa cieco. Per leggere il
log: `tr '\r' '\n' < logs/run.log | tail -20`.

Per ispezionare il risultato senza aprire un player:

```sh
ffmpeg -v error -i outputs/nome.mp4 \
  -vf "select='eq(n\,0)+eq(n\,18)+eq(n\,37)+eq(n\,55)',tile=4x1" \
  -frames:v 1 -y outputs/contact.png
```

Poi leggi il PNG. Serve a verificare che identità e stile non derivino.

## Nota sul disco

Durante un run lo spazio libero oscilla di ~6 GiB e torna su da solo: è spazio
purgeable APFS delle letture uncached, non una perdita. Non allarmarti e non
metterti a cercare file. Verificato campionando `df` ogni 5 s per l'intero run.
