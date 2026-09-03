# SPEC.md — LoRA dinamici in h3-metal

## 0. Header

**Fonti consultate**

| Fonte | Cosa ne viene |
|---|---|
| Sessione di lavoro del 2026-09-01/02 con NeoKree | scope, le quattro decisioni di Fase 3, il caso d'uso turbo LoRA |
| Codice `h3.c` allo stato del commit `8974cc0` | punti di innesto, vincoli di percorso, infrastruttura di test |
| `README.md` del repo | comportamento documentato di `--ssd-streaming` e dello schedule |
| `minimax_h3_turbo_v4_step600_pruned_comfyui.safetensors` | schema del file LoRA, verbatim in sezione 7 |
| Header di `larryvrh/MiniMax-H3-Turbo-Lora` / `minimax_h3_turbo_4step.safetensors` (779,8 MB, letto con range request da 1 MB) | prefisso opzionale, ranghi non uniformi, chiavi AdaLN verbatim, il bersaglio `final_layer` |
| `H3_Combat_V2.safetensors` (147,9 MB, ai-toolkit) | secondo LoRA reale a rango 16, corpus di test in sezione 8bis |
| Lettura di `llama-adapter.cpp` / `llama-graph.cpp` / `llama-context.cpp` come arte nota (`docs/research/llama-cpp-adapters.md`, branch `research/llama-cpp-adapters`) | politica di fallimento, rango per coppia, aggiunte al piano di test |

**Destinazione di questo documento**: il lavoro di pianificazione in corso produce
un **design chiuso** — ogni punto aperto reso non ambiguo. L'implementazione è una
consegna successiva, **non** fa parte di questa mappa.

**Disciplina**: ogni punto di questo documento è un **requisito fisso**, non un'ipotesi.
Dove il codice è citato con un numero di riga, quel riferimento è stato verificato
con una ricerca diretta nel sorgente durante la stesura. Tutto ciò che non è stato
deciso sta in **sezione 10** e non va risolto con assunzioni silenziose.

---

## 1. Obiettivo

Permettere a `h3` di applicare uno o più LoRA **a runtime**, senza fondere i pesi
nel checkpoint e senza modificare il modello base su disco.

---

## 2. Attori e autenticazione

Non c'è autenticazione: `h3` è un binario CLI locale che gira per conto
dell'utente che lo invoca. Gli attori sono due, e usano la stessa libreria:

- **Utente interattivo** — sessione `h3` senza `-p`, comandi `!`.
- **Utente one-shot / programma chiamante** — `h3 -p ...`, oppure `libh3.a`.

---

## 3. Flusso operativo

1. L'utente indica uno o più file LoRA, con una `strength` per ciascuno.
2. `h3` legge l'header safetensors di ogni file, **riconosce la convenzione di
   naming**, e normalizza le coppie in una rappresentazione interna unica.
3. Per ogni coppia, `h3` verifica che il tensore bersaglio esista nel checkpoint
   e che le forme siano compatibili. **Una coppia il cui nome non ha un bersaglio,
   o la cui forma non entra in un bersaglio supportato, è fatale**: il caricamento
   si ferma, il messaggio d'errore le elenca **tutte** (stringhe alla sezione
   5bis.2), e nessuna generazione parte. Sono fatali anche un file illeggibile o non parsabile e una coppia
   internamente incoerente, cioè con il rango di `A` e quello di `B` discordi.

   Questo punto è stato rivisto due volte. La formulazione originale ("una
   incompatibilità è un errore, non un avviso") era stata sostituita da una
   politica di segnalazione e salto, ritirata il 2026-09-03. Motivo del ritorno al
   fatale: il salto obbliga il report a portare un canale di comunicazione intero
   (quante coppie, raggruppate per quale motivo, quante mostrarne prima di
   inondare il terminale) per un caso che nessun file del corpus produce. Le 208
   coppie del LoRA di riferimento trovano tutte un bersaglio. Rifiutare il file
   intero costa all'utente un messaggio invece di un video sbagliato, e toglie dal
   report la sua sezione più complicata.

   Il prezzo, dichiarato: un file con **una sola** coppia verso un tensore che
   `h3` non implementa è inutilizzabile finché quel bersaglio non viene aggiunto.
   Rispetto a `llama.cpp` la divergenza resta, spostata: lì l'abort riporta **solo
   la prima** coppia orfana (`llama-adapter.cpp:331`) e le coppie `_norm.weight`
   sono saltate in silenzio sotto un TODO (`:287-290`). Qui l'errore le elenca
   tutte e non ne salta nessuna.
4. Gli adapter restano residenti in memoria come tensori separati. **I pesi base
   non vengono mai modificati**, né su disco né nei buffer GPU.
5. A ogni valutazione del denoiser, per ogni proiezione bersaglio, la GPU calcola
   `y = Wx + strength * B(Ax)` invece di `y = Wx`.
6. In sessione interattiva l'utente può aggiungere, rimuovere o riscalare un LoRA
   senza ricaricare il modello; la generazione successiva usa il nuovo insieme.
   **Qualificazione** (vedi sezione 7bis): il LoRA si applica anche al token refiner
   e all'AdaLN, i cui risultati sono precalcolati una volta sola. Un cambio
   dell'insieme LoRA attivo **invalida e ricalcola** `refined_text` e lo schedule
   AdaLN. Non è quindi hot-swap "senza ricaricare il modello" in senso stretto:
   è un **ricalcolo parziale**, i 62 GiB di pesi base restano dove sono.

---

## 4. Regole di business vincolanti

**H1 — Il checkpoint base è immutabile.** Nessuna scrittura, nessun `add_inplace`
sui tensori del modello, nessun file derivato. Questa è la ragione d'essere del
progetto: l'alternativa è fondere i pesi offline e riscrivere il checkpoint,
che è esattamente ciò che qui si rifiuta di fare.

**H2 — L'adapter è un ramo separato, non una fusione in memoria.** Deve valere
anche mentre i pesi vengono riletti dall'SSD a ogni valutazione. Qualsiasi
soluzione che scriva il delta dentro il buffer del peso è esclusa per costruzione:
con `--ssd-streaming` quel buffer viene sovrascritto dalla lettura successiva.

**H3 — Equivalenza numerica dimostrata.** Per ogni proiezione bersaglio,
`Wx + strength * B(Ax)` deve coincidere con `(W + strength * B@A)x` entro la
tolleranza BF16 dichiarata in sezione 8. Non "sembra uguale": misurato.

**H4 — `strength = 0` è identità.** Con tutti i LoRA a zero, o senza LoRA,
l'output deve essere **byte-identico** a quello di `h3` prima della modifica,
a parità di seed e parametri. Nessuna regressione sul percorso senza LoRA.

**H5 — Nessun bersaglio ignorato in silenzio.** Se un file LoRA contiene coppie
che puntano a tensori che `h3` non applica, il caricamento **fallisce** e l'errore
le elenca tutte. Né il salto né il silenzio sono consentiti (sezione 3, punto 3).
`llama.cpp` salta le coppie `_norm.weight` senza dire niente, sotto un TODO
(`llama-adapter.cpp:287-290`): H5 vieta esattamente questo.

Caso distinto, che H5 **non** copre: le coppie che nel file non ci sono. Un turbo
LoRA a cui il convertitore ha tolto metà degli adapter non porta coppie orfane,
ne porta di meno, e non fa scattare nessun errore. A dirlo è il conteggio AdaLN
sempre dichiarato più l'avviso sulla firma di conversione (sezione 5, punto 1):
è esattamente il modo in cui il file di prova di questa sessione sarebbe passato
inosservato.

**H6 — La scala dichiarata dal file vince.** Nelle convenzioni che portano
`alpha`, il fattore effettivo è `alpha/rank` e va letto dal file, non assunto.
La `strength` dell'utente moltiplica quel fattore, non lo sostituisce.

Uso reale della `strength`, dichiarato dal proprietario: **con segno**, in
`[-2, 2]`, e nel 90% dei casi fra 0,6 e 0,7. La strength 100 di T1 (sezione 8) è
quindi uno strumento di misura 50 volte fuori dall'intervallo di produzione, non
un regime supportato. **Dominio della `strength`: qualunque valore finito, nessun estremo.** Nessun
tetto superiore e nessun pavimento inferiore. Un tetto obbligherebbe a scegliere
un numero indifendibile fra la produzione (2) e T1 (100), e un valore enorme non
rompe niente: bf16 ha lo stesso range di esponente del float32, quindi non c'è
overflow, esce solo un video sbagliato. Nemmeno un avviso sopra una soglia:
l'errore realistico è `:10` al posto di `:1.0`, che sta a 5x dalla produzione,
quindi una soglia bassa abbastanza da prenderlo urlerebbe su ogni run di T1, e
una alta abbastanza da tacere su T1 non lo prende mai. La strength effettiva è
già stampata nel report di attivazione (sezione 5), che è dove l'utente verifica
cosa ha chiesto.

**Si rifiuta solo ciò che non è un numero finito, e in due punti.**

Al **parse della CLI**, su entrambe le superfici: spazzatura in coda
(`:1.0abc`), stringa vuota, e `isfinite` falso (`nan`, `inf`, `-inf`). Il parse
deve essere rigido e non indulgente: `atof("abc")` ritorna `0.0`, e la strength 0
fa scartare l'adapter alla costruzione dell'insieme, quindi un parse permissivo
trasformerebbe un refuso in una **LoRA sparita in silenzio**, che è esattamente
ciò che H5 vieta. La reazione segue la convenzione della repo: one-shot stampa
`h3: invalid ...` ed esce con `2` (`parse_int`, `main.c:66`), l'interattiva
stampa e lascia il valore precedente (`parse_i32`, `h3_cli.c:58`). Dominio
identico sulle due superfici, reazione diversa: non è una divergenza fra
`--lora` e `!lora set`, è come si comportano già tutte le altre opzioni.

Alla **costruzione dell'insieme attivo**, solo `isfinite`, e il fallimento è
quello di una coppia fatale (sezione 3, punto 3). Non è ridondanza: `h3_params`
è API pubblica (sezione 6ter) e un embedder può scriverci un `NaN` senza passare
dalla CLI. `strtof("nan")` **riesce**, quindi la rigidità sintattica non lo
prende; e non lo prende nemmeno lo scarto a strength 0, perché `NaN != 0`. Un
`NaN` viene fuso dentro `A` (sezione 7bis.4), rende `NaN` l'intero delta e
produce un video nero senza un messaggio. Precedente in casa:
`frames_from_seconds` (`main.c:82`) fa già esattamente questo controllo.

**La coda dopo l'ultimo `:` che non è un numero è un errore, non un percorso.**
G3 divide sull'ultimo due punti, e su macOS i due punti sono legali nei nomi di
file, quindi `refs/turbo:v2.safetensors` è un percorso possibile. Non si ricade
sul trattare l'intera stringa come percorso a strength `1.0`: quel fallback
renderebbe `file.safetensors:0.8x` un percorso inesistente, e l'errore mostrato
diventerebbe "file non trovato" invece di "strength malformata", cioè punterebbe
nel posto sbagliato. Il messaggio nomina il pezzo che non ha parsato, così anche
chi ha davvero un due punti nel nome capisce che deve scrivere `:1.0`.

**`strength 0` resta scartata, ma dichiarata** (sezione 5, punto 1). Lo scarto
alla costruzione dell'insieme è ciò che rende H4 strutturale e non si tocca.

---

## 5. Reportistica

Non c'è dashboard né export. Gli output secondari richiesti sono tre, tutti
testuali:

1. **All'attivazione di un LoRA**: percorso, **distribuzione dei ranghi** (non un
   rango unico, vedi 7.2), numero di coppie applicate, **numero di coppie AdaLN**,
   memoria occupata. Non esiste una sezione "coppie saltate": una coppia
   inapplicabile è fatale (sezione 3, punto 3) e il suo elenco esce dal messaggio
   d'errore, non dal report.

   Un `--lora` a **strength 0** non sparisce dal report. L'adapter viene scartato
   alla costruzione dell'insieme attivo, ed è così che H4 diventa strutturale, ma
   il report ne porta comunque una riga: `stile.safetensors: strength 0, not
   applied`. È l'unico caso in cui un LoRA richiesto esplicitamente potrebbe non
   esserci senza che nessuno lo dica; il file viene validato lo stesso, quindi un
   percorso sbagliato a strength 0 resta un errore e non un silenzio.

   **Forma della distribuzione dei ranghi**: istogramma compatto su una riga,
   ordinato per conteggio decrescente, `ranks: 64 x208, 16 x51`. Non un intervallo
   (`16-64` nasconde la bimodalità, che è il segnale utile: 16 è l'AdaLN, 64 il
   backbone) e non un raggruppamento per tipo di bersaglio, che inventerebbe una
   tassonomia da mantenere a ogni bersaglio nuovo.

   Il conteggio delle coppie AdaLN comprende `final_layer.adaln_proj.linear`
   insieme ai 50 `blocks.N.adaln_proj.linear`, quindi su un turbo upstream
   completo vale **51 e non 50**. È lo stesso numero che un file potato cita in
   `removed_pair_count=51`: contarne 50 metterebbe due cifre diverse per la
   stessa cosa a due righe di distanza.

   Il conteggio delle coppie AdaLN va dichiarato **sempre**, anche quando è zero.
   Zero coppie AdaLN è normale in un LoRA di stile e patologico in un LoRA turbo,
   e solo l'utente sa quale dei due sta caricando. In più, se l'header del file
   porta una firma di conversione (`partial_conversion`, `removed_pair_count`,
   `adaln_keys_removed`), va emesso un avviso esplicito che la cita: quel file è
   stato potato da un convertitore e il numero di coppie mancanti è scritto lì.

   **Come si distinguono i due**: per prefisso e posizione di riga, non per
   colore. Il conteggio è un campo della riga di riepilogo normale
   (`AdaLN pairs: 51`); l'avviso è una **riga separata che inizia con `warning:`**
   e **cita i valori dell'header verbatim** (`removed_pair_count=51`,
   `partial_conversion=true`) invece di parafrasarli. Il file sa più di `h3` su
   cosa gli è stato tolto, e riscriverlo a parole perde il numero.
2. **`!status`** elenca i LoRA attivi con la rispettiva strength su **una riga
   sola**, con il basename e non il percorso intero:
   `LoRA: style.safetensors 0.80, turbo.safetensors 1.00`, e `LoRA: none` quando
   l'insieme è vuoto. Il precedente in casa è `!refs`, che stampa un conteggio e
   rimanda a un comando dedicato (`h3_cli.c:206`). Il report completo del punto 1
   esce **una volta sola all'attivazione**, cioè su `!lora add` attraverso
   `h3_lora_preload`, che risponde nell'istante in cui lo digiti perché legge solo
   gli header (sezione 6ter, punto 3). `!lora` nudo ripete l'elenco lungo su
   richiesta.
3. **`--profile`** — requisito **ritirato**. L'API di profiling del repo è
   due funzioni (`h3_gpu.h:98-99`), `h3_gpu_profile_set_label` e
   `h3_gpu_profile_mark(gpu, phase)`: marcatori di fase grossolani, nient'altro.
   Il ramo LoRA è interleaved dentro la fase di denoise, più volte per blocco per
   step: non esiste un confine di fase da marcare, e costruire un timing
   per-kernel sarebbe un progetto di profiling, non una feature LoRA.
   `--profile` continua a contare tutto insieme; il costo del LoRA si misura
   **fuori banda**, con run A/B, quando diventerà rilevante (sezione 10, G6).

---

## 5bis. Messaggi d'errore

Le stringhe dei casi fatali e dello stop del guardrail sono contratto come lo è
il report: H5 esige che l'utente veda cosa è mancato, e a tredici minuti per run
un messaggio vago costa un run.

### 5bis.1 Il canale: riepilogo su `h3_last_error`, dettaglio sul callback

`h3_set_error` è una `vsnprintf` su **512 byte fissi** (`h3.c:361-366`,
`h3_internal.h:14`). Le 208 coppie che H5 pretende elencate non ci entrano, e
nemmeno tre con le shape. Il messaggio fatale ha quindi **due parti**:

- **Una riga di riepilogo** in `h3_last_error`, autosufficiente: chi integra
  `libh3.a` senza installare un callback deve comunque sapere cosa è mancato e
  quanto.
- **Le righe di dettaglio** attraverso `h3_report_callback`, una per coppia,
  senza cap.

Conseguenza sull'API: **`h3_params` guadagna `on_report`** accanto a `on_frame`
e `on_progress` (`h3.h:126-128`). Senza, un embedder che salta
`h3_lora_preload` riceve il fallimento di `h3_generate` (sezione 6ter punto 6)
ma non l'elenco, e resta con il solo numero. La libreria **non stampa mai su
`stderr` da sé**.

Le righe del callback **non portano il prefisso `h3: `**: lo mette la CLI
stampandole, su **ogni** riga e non solo sulla prima, perché è quello che fa
sopravvivere un elenco all'interleaving con l'avanzamento scritto in `\r` sullo
stesso `stderr`.

### 5bis.2 Coppie inapplicabili

Riepilogo, con il **percorso come l'utente l'ha scritto** e il conteggio
**spaccato per tipo**:

```
loras/style.safetensors: 3 pairs cannot be applied (2 with no target, 1 with a shape mismatch)
```

Il percorso verbatim è una divergenza deliberata dal basename della sezione 5
punto 2: nel report il file è già identificato dal contesto, in un errore la
stringa utile è quella che l'utente deve correggere, e due `style.safetensors` in
due cartelle sono un caso reale. Lo spacco per tipo c'è perché quella riga è
**tutta** la diagnosi di chi non ha installato `on_report`.

Dettaglio: **lista piatta**, ordine dell'header del file, tipo come tag di riga.

```
style.safetensors: 3 unapplicable pairs, load aborted
  no target: blocks.51.attn.qkv (rank 64)
  no target: blocks.51.mlp.fc1 (rank 64)
  shape mismatch: blocks.0.mlp.fc1 is [64 x 4096], target is [64 x 3584]
```

Piatta e non in due sezioni: i due tipi non sono azionabili separatamente,
entrambi dicono "questo file non è per questo modello" e il rimedio è lo stesso,
mentre due sezioni obbligherebbero a riconciliare due conteggi contro il
riepilogo. Il nome è quello della **coppia**, senza il suffisso
`lora_A`/`lora_B`, perché la coppia è l'unità. Il **rank** su `no target` perché
16 contro 64 dice subito se è una coppia AdaLN o backbone; le **due shape** su
`shape mismatch` perché la differenza è la diagnosi.

**Nessun cap, nessuna coda "and N more"**: è la ragione per cui il dettaglio va
sul callback e non nei 512 byte. Il caso a 208 righe è il file sbagliato per il
modello, dove l'elenco intero *è* la diagnosi; il caso realistico è 1-5, e le 208
coppie del file di riferimento trovano tutte un bersaglio.

### 5bis.3 Convenzione B rifiutata

```
turbo.safetensors: lora_up/lora_down naming is not supported
  (first seen at blocks.0.attn.qkv.lora_up.weight); h3 reads lora_A/lora_B
```

**Un solo offendente**, citato come prova. La convenzione è una proprietà del
**file**, non della coppia: elencare 208 `lora_up` non dice niente più di uno. La
distinzione da tenere: *coppia inapplicabile è per-coppia, quindi tutte;
convenzione è per-file, quindi una*. Il messaggio nomina la convenzione che `h3`
legge perché il rimedio è convertire il file, e senza quel pezzo l'utente non sa
verso cosa convertirlo. Non `unsupported format`, che invita una bug report
invece di una conversione.

### 5bis.4 Strength malformata

I due punti di rifiuto sono fissati sotto H6: il parse della CLI e `isfinite`
alla costruzione dell'insieme attivo.

```
h3: invalid --lora strength: 0.8x (in loras/style.safetensors:0.8x)
lora: invalid strength: 0.8x; keeping 0.80
lora: strength for style.safetensors is not a finite number
```

La prima segue lo stile di casa alla lettera (`main.c:71,100`, `h3: invalid %s:
%s` più `exit(2)`) e aggiunge l'argomento intero fra parentesi, che è l'unica
cosa che disambigua fra cinque `--lora` sulla stessa riga. La seconda è
`!lora set`, che per convenzione tiene il valore precedente: **deve dire quale
valore ha tenuto**, altrimenti l'utente crede di aver cambiato qualcosa. La terza
è la rete `isfinite` sull'API pubblica, dove non c'è un frammento di testo da
citare perché il `NaN` arriva come `float`.

### 5bis.5 Lo stop del guardrail

```
h3: not enough memory for this run: 34.1 GiB needed, 32.0 GiB available
    (Metal working set 36.0 GiB minus the 4 GiB reserve).
    Lower the canvas or drop an adapter.

h3: memory ceiling hit after the first denoiser evaluation:
    footprint 30.2 GiB, ceiling 27.4 GiB (system free 24.9 GiB plus our 2.5 GiB).
    Another process is holding memory: quit it, or lower the canvas.
```

Il messaggio **deve dire quale dei due termini del `min` ha morso** (sezione 10,
G4), perché il rimedio cambia: il termine statico vuole un canvas più piccolo,
quello dinamico vuole che l'utente chiuda il processo co-residente. Il termine si
nomina con **la cosa fisica** (working set di Metal, memoria libera di sistema) e
non con i nostri nomi di progetto: "statico" e "dinamico" non dicono all'utente
cosa toccare.

I gate **non si numerano** nel messaggio: `gate 1` e `gate 2` sono vocabolario
nostro. Il messaggio dice **quando** è scattato, e `after the first denoiser
evaluation` è già l'informazione utile perché distingue il fallimento in un
secondo da quello dopo un'evaluation. Entrambe le stringhe stanno nei 512 byte,
quindi escono da `h3_last_error` senza toccare il callback, che sul percorso del
gate 2 (dentro `h3_generate`) potrebbe non esserci installato.

### 5bis.6 One-shot e interattivo

**Le stringhe della libreria sono identiche sulle due superfici.** Cambia la
reazione, com'è già la convenzione di casa per ogni altra opzione: one-shot esce,
interattivo tiene l'insieme attivo precedente. In interattivo la **CLI** aggiunge
una riga sua:

```
h3: the active set is unchanged
```

Senza, l'utente non sa se l'`!lora add` è atterrato a metà. La stampa la CLI e
non la libreria: è un fatto della sessione, e tenere la libreria agnostica alla
superficie è quello che rende le stringhe identiche in primo luogo.

---

## 6. Stack tecnologico e ambiente

- **Linguaggio**: C (C11), come il resto del repo. Kernel in Metal Shading
  Language dentro `h3_shaders.metal`.
- **Nessuna dipendenza nuova.** Il repo oggi non ne ha, e non le acquisisce
  per questa feature.
- **Parsing safetensors**: si riusa `h3_safetensors.c`, che già esiste.
- **Hardware di riferimento**: Apple M4 Pro, 48 GiB unified, `Metal 4: no`.
  Le ottimizzazioni marcate M5 nel README non si attivano su questa macchina e
  non fanno parte del criterio di accettazione.
- **Build**: `make -j8`, target esistenti. Ogni nuovo test è un target nel
  `Makefile` accanto agli altri.

---

## 6bis. Struttura repo

Il codice nuovo vive accanto all'esistente, senza cartelle nuove:
`h3_lora.c` / `h3_lora.h` per il caricamento e la rappresentazione, innesti
puntuali in `h3_dit.c`, opzione in `main.c`, comando in `h3_cli.c`.

Nessuno script di supporto in Python: la directory `tools/` è stata eliminata e
non è mai stata committata. L'oracolo di sezione 8 è **dentro il binario di test
in C**, non fuori.

### Arte nota

`ggml-org/llama.cpp` e `leejet/stable-diffusion.cpp` sono stati letti come
precedenti, non come modelli da imitare: le loro scelte compaiono in questo
documento solo dove servono a giustificare una nostra decisione, con la riga di
codice accanto. Le due che contano: `stable-diffusion.cpp` somma il delta dentro
il tensore base (`src/model/adapter/lora.hpp:937`, `ggml_add_inplace`), che H2
esclude; `llama.cpp` lo somma sull'attivazione, che è la forma compatibile con
lo streaming. Lettura completa in `docs/research/llama-cpp-adapters.md`.

---

## 6ter. Superficie API

Il contratto pubblico è **solo** `h3_params` più due funzioni. Non esiste un tipo
opaco che il chiamante possieda: gli adapter vivono nella cache di `h3_ctx`.

1. **Dichiarativa, non a handle.** `h3_params` porta
   `const h3_lora *loras; size_t lora_count;` con
   `h3_lora = { const char *path; float strength; }`, nello stesso modo in cui
   `h3_reference` porta già un path. Niente da liberare per il chiamante,
   quindi la domanda "un adapter può sopravvivere al modello" non si pone.
2. **`h3_lora.h` è interno**, non fa parte del contratto di `libh3.a`. La
   superficie pubblica minima lascia libere le mani a G4.
3. **Validazione contro gli header, subito.**
   `h3_lora_preload(ctx, path, cb, opaque)` confronta l'header safetensors del
   LoRA con gli header degli shard del transformer, **senza leggere un byte di
   peso**. Il report esce quindi prima dei 62 GiB, e `!lora add` risponde
   nell'istante in cui lo digiti invece che alla generazione successiva.
4. **Report su callback.**
   `typedef void (*h3_report_callback)(const char *line, void *opaque)`, la
   forma di `h3_progress_callback` **tranne il ritorno**. La libreria non stampa
   da sé: chi integra `libh3.a` decide dove finisce il testo, e H5 esige che si
   veda. Il callback vive su `h3_params` accanto a `on_frame` e `on_progress`
   (campo `on_report`) oltre che come argomento di `h3_lora_preload`.

   **Il ritorno è `void`, e la ragione va tenuta**: un `int` non-zero da
   `h3_progress_callback` significa cancel e fa
   `h3_set_error(ctx, "generation cancelled during %s", phase)`
   (`h3.c:648-652`), che **sovrascrive** l'errore già impostato. Copiando quella
   forma, un callback che ritorna non-zero a metà elenco cancellerebbe il
   messaggio fatale e l'utente leggerebbe `cancelled` invece della diagnosi. Un
   report non ha poi niente da annullare: le righe sono il resoconto di una
   decisione già presa, e "annullare a metà elenco" non ha un significato. Un
   `int` ignorato sarebbe peggio, perché il prossimo lettore lo crederebbe
   significativo.
5. **Chiave di cache: path più dimensione più mtime** (`stat`). Un file
   sovrascritto allo stesso path viene ricaricato invece di restare quello
   vecchio in memoria senza dirlo.
6. **Un caso fatale ferma la generazione.** `h3_generate` fallisce, il
   **riepilogo** di una riga esce da `h3_last_error` e il **dettaglio** esce
   riga per riga da `on_report`, perché `h3_set_error` scrive in 512 byte fissi
   e l'elenco che H5 pretende non ci sta (sezione 5bis.1). Non si consegna mai
   un video privo del LoRA richiesto: a tredici minuti per run, non c'è modo di
   accorgersene guardandolo. Vale solo per i due casi fatali di sezione 3
   punto 3.
7. **`h3_lora_release(ctx, path)`** libera una singola voce di cache, che è
   quello che serve a `!lora remove` senza svuotare anche il DiT preparato.
8. **L'insieme attivo si sostituisce in blocco** a ogni generazione, perché vive
   in `h3_params`. `!lora add|set|remove` è editing della lista nello stato
   della sessione (`h3_cli.c:28`, copiata per generazione a `h3_cli.c:434`), non
   una API incrementale. Una strength nuova non ricarica niente: la porta il
   params, l'adapter in cache non la conosce.

Due conseguenze da non perdere:

- **Su H4**: le voci a strength 0 vanno scartate mentre si costruisce l'insieme
  attivo, così il loop di dispatch è identico a quello senza LoRA **per
  costruzione** e non per fortuna.
- **Su 7bis.2 e 7bis.3**: il trigger di ricalcolo di refiner e schedule AdaLN è
  "la lista LoRA di questo `h3_params` differisce da quella con cui la cache è
  stata calcolata". Il dettaglio è deciso a parte.

---

## 7. Modello dati

Contratto intoccabile. Nomi e forme rilevati dal file di prova e dal checkpoint
FL2VA, non inventati.

### 7.1 Tensori bersaglio nel checkpoint

Quattro proiezioni per blocco, già enumerate dal codice in **due** punti:

```c
/* h3_dit.c:517-522 — percorso a pesi residenti */
LOAD2(qkv, "attn.qkv_proj.weight", INNER * 3, HIDDEN);
LOAD2(out, "attn.out_proj.weight", HIDDEN, INNER);
LOAD2(fc1, "mlp.fc1.weight",       FFN * 2,   HIDDEN);
LOAD2(fc2, "mlp.fc2.weight",       HIDDEN,    FFN);

/* h3_dit.c:615-618 — percorso --ssd-streaming */
SOURCE(0, "attn.qkv_proj.weight", INNER * 3, HIDDEN, STREAM_QKV);
SOURCE(1, "attn.out_proj.weight", HIDDEN,    INNER,  STREAM_OUT);
SOURCE(2, "mlp.fc1.weight",       FFN * 2,   HIDDEN, STREAM_FC1);
SOURCE(3, "mlp.fc2.weight",       HIDDEN,    FFN,    STREAM_FC2);
```

I blocchi del token refiner passano dalla **stessa** `load_block`
(chiamata a `h3_dit.c:819-821`, definizione a `h3_dit.c:502`), quindi un innesto
unico li copre — ma girano una volta sola per generazione: vedi sezione 7bis.2.

Forme reali, lette dagli header dei shard FL2VA:

| Tensore | Forma |
|---|---|
| `blocks.N.attn.qkv_proj.weight` | `[21504, 5376]` |
| `blocks.N.attn.out_proj.weight` | `[5376, 7168]` |
| `blocks.N.mlp.fc1.weight` | `[28672, 5376]` |
| `blocks.N.mlp.fc2.weight` | `[5376, 14336]` |

Copertura delle sole quattro proiezioni: **208 tensori su 535, pari al 60,5%
del DiT in byte** (37,3 GiB su 61,7).

**Ci sono anche due bersagli AdaLN.** Non sono proiezioni per-step ma precalcoli
una tantum (vedi 7bis.3), e questa sezione non li enumerava perché è stata
scritta leggendo il file di prova potato, che non ne porta nessuno:

| Tensore | Forma | Caricato in |
|---|---|---|
| `blocks.N.adaln_proj.linear.weight` | `[96768, 2688]` | `h3_dit_schedule.c:267` |
| `final_layer.adaln_proj.linear.weight` | `[10752, 2688]` | `h3_dit_schedule.c:304` |

`final_layer` è **una** coppia sola, non una per blocco: è il layer di uscita, e
le 51 coppie AdaLN del turbo upstream sono 50 blocchi più questa. Il bersaglio si
risolve per nome contro il checkpoint, che è come il loader scopre già la forma,
quindi non serve un ramo in più nel controllo: serve un nome in più
nell'enumerazione.

Copertura con gli AdaLN: **259 tensori su 535, pari al 99,8% del DiT in byte**
(61,60 GiB su 61,73). Misurato sommando gli header dei 13 shard FL2VA.

### 7.2 Schema del file LoRA

Convenzione A, rilevata nel file di prova:

```
diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight   BF16  [64, 5376]
diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight   BF16  [21504, 64]
diffusion_model.token_refiner.blocks.0.mlp.fc1.lora_A.weight   BF16 [64, 5376]
```

Nome del bersaglio = chiave meno il suffisso `.lora_{A,B}.weight`, più
`.weight`, e meno il prefisso `diffusion_model.` **se presente**.

**Il prefisso `diffusion_model.` è opzionale.** Il file di prova, convertito per
ComfyUI, lo porta; il turbo LoRA upstream non pruned **non lo porta**. Un loader
che lo rimuove senza verificare che ci sia non trova **nessuna** coppia su
quest'ultimo. Vanno accettate entrambe le forme.

**Il rango non è uniforme dentro un file.** Nel turbo upstream le coppie AdaLN
sono a rango 16 e tutte le altre a rango 64. Non esiste quindi un "rango del
file": il rango si legge **per coppia**, dalla forma di `A` (`[rank, in]`) e di
`B` (`[out, rank]`), e i due valori devono concordare. Se discordano, la coppia è
internamente incoerente e il caso è fatale (sezione 3, punto 3). Anche `llama.cpp`
legge il rango per coppia, da `b->ne[0]` (`llama-adapter.cpp`).

Metadati del file di prova, verbatim:

```json
{"application": "W_eff = W + lora_B @ lora_A",
 "base_model": "MiniMax-H3",
 "retained_pair_count": "208",
 "removed_pair_count": "51",
 "partial_conversion": "true",
 "warning": "All AdaLN LoRA adapters were removed because the source targets
             input dimension 2688 while the pruned model targets dimension 8.
             Four-step distillation behaviour may be degraded or broken."}
```

Quel `warning` è il motivo per cui esistono il conteggio AdaLN sempre dichiarato
e l'avviso di conversione (sezione 5, punto 1). Le 51 coppie potate **non sono
orfane, sono assenti**: nessun controllo di forma le vede, quindi non fanno
scattare il fatale di H5. Solo l'header del file sa che c'erano.

### 7.3 Convenzioni: cosa si supporta e cosa si rifiuta

**Supportata**: la convenzione A qui sopra, `lora_A` / `lora_B`, **più la
variante ibrida** in cui le stesse coppie sono accompagnate da un tensore
`.alpha`. In quel caso il fattore di scala è `alpha/rank` letto dal file
(regola H6), con il rango **della coppia**, non del file.

**Rilevata e rifiutata**: la convenzione B, `lora_up` / `lora_down` con i nomi
appiattiti a underscore, viene **riconosciuta e respinta con un errore chiaro**.
Non è supportata. Motivo: nessun file nel corpus del proprietario la usa, e
supportarla vorrebbe dire scrivere e mantenere un de-appiattimento dei nomi senza
un solo caso di prova su cui validarlo. L'errore deve nominare la convenzione
rilevata, così che l'utente sappia che il file non è corrotto ma solo nel formato
sbagliato.

Questa decisione **sostituisce** quella di Fase 3 ("convenzione B da supportare
per rilevamento automatico").

**Rilevata e rifiutata**: gli **adapter di bias**, tensori `.diff_b`. Sono una
terza categoria, né una coppia low-rank né un bersaglio mancante: un *tipo* di
adapter che h3 non implementa. Un `.diff_b` è un delta a rango pieno sul bias,
`[96768]` F32, uno per blocco, che si sommerebbe a
`blocks.N.adaln_proj.linear.bias` (bersaglio che nel checkpoint **esiste**, con
esattamente quella forma). Verificato nell'header di
`barelymining/ComfyUI-MiniMax-H3-FastVideo/fasth3_vsa_4-steps-v5.safetensors`:
615 tensori, 150 chiavi AdaLN = 50 blocchi x (`lora_A`, `lora_B`, `diff_b`).

Il messaggio deve dire che gli adapter di bias non sono supportati, **non**
"suffisso inatteso": il file non è malformato, contiene una cosa che sappiamo
riconoscere e abbiamo scelto di non applicare.

Motivo del rifiuto invece del supporto: implementarlo non sbloccherebbe **nessun
file reale conosciuto**. Quello stesso file cade comunque prima, sul controllo di
forma AdaLN (`lora_A [64, 8]` da `B@A = [96768, 8]` contro `[96768, 2688]`), che
è il percorso fatale della §3.3. Il bias delta è il secondo ostacolo, non il
primo. Il costo sarebbe modesto e non violerebbe H1 (il bias è già un tensore a
parte passato a `h3_gpu_linear_bf16` in `h3_dit_schedule.c:290` e liberato a
`:294`, quindi `bias + s*diff_b` starebbe in uno scratch ricostruito a ogni load,
mai in un buffer streamato) ma comprerebbe zero file utilizzabili in più.

Precedente: `llama.cpp` lancia su suffisso inatteso
(`src/llama-adapter.cpp:291`). Qui si fa lo stesso, nominando la categoria.

Se comparirà un file il cui **unico** ostacolo è il `.diff_b`, l'errore lo dirà
per nome e la decisione si riapre allora, con un caso di prova in mano.

---

## 7bis. Percorsi che il LoRA attraversa davvero

Tre fatti di percorso, verificati sul sorgente, che cambiano la forma dell'innesto.

### 7bis.1 Il kernel QKV fuso è codice morto su questa macchina

`h3_gpu_grouped_qkv_linear_rope_bf16` (`h3_gpu.m:3780`) prende il suo percorso
NAX fuso **solo** se `gpu.tensorOpsEnabled`, e quel flag richiede che il nome del
device Metal contenga la stringa letterale `"M5"` (`h3_gpu.m:364-373`). La
macchina di riferimento è un M4 Pro: la funzione cade **sempre** sul fallback,
cioè `h3_gpu_linear_bf16(...)` seguito da `h3_gpu_grouped_qkv_rope_bf16(...)`
(`h3_gpu.m:3805-3810`).

Conseguenza operativa: `dit->qkv` è quindi la proiezione **grezza**, non normata
e senza RoPE, e il delta LoRA per qkv va inserito **fra le due chiamate**.

Modificare il kernel NAX fuso perché trasporti anche gli argomenti LoRA è
**fuori scope** per questo giro: è solo-M5, e la sezione 6 già esclude le
ottimizzazioni M5 dal criterio di accettazione.

### 7bis.2 Il token refiner gira una volta sola

`token_refiner.blocks.{0,1}` passano da `load_block` (`h3_dit.c:502`) dentro
`refine_text` (`h3_dit.c:804-880`). Quella funzione gira **esattamente una volta
per generazione**, chiamata a `h3_dit.c:1667` prima del loop del denoiser, e
libera i pesi dei suoi blocchi subito dopo (`h3_dit.c:869`); il suo output è
cachato in `dit->refined_text` e riusato da ogni step.

Decisione: **il LoRA si applica al refiner**, e un cambio dell'insieme LoRA attivo
deve invalidare e ricalcolare `refined_text`. È la qualificazione del punto 6
della sezione 3.

### 7bis.3 AdaLN è lo stesso problema, non un quinto tensore

`blocks.N.adaln_proj.linear.weight` è caricato in `h3_dit_schedule.c:267` e
consumato da `h3_dit_schedule_precompute` (`h3_dit.c:1670`, etichetta di
avanzamento "precompute AdaLN" a `h3_dit.c:1556`). La modulazione è precalcolata
una volta per tutti gli step e cachata; il denoiser si limita a leggere
`h3_dit_schedule_block()`.

C'è un secondo tensore con lo stesso ruolo e un solo esemplare,
`final_layer.adaln_proj.linear.weight` `[10752, 2688]`, caricato sette righe più
sotto a `h3_dit_schedule.c:304`. È enumerato in 7.1 insieme all'altro.

`adaln_proj` **non** è quindi una quinta proiezione per blocco accanto alle
quattro della sezione 7.1: strutturalmente è lo stesso problema del refiner, un
precalcolo una tantum invalidato da un cambio di LoRA, **non** una proiezione
per-step in streaming. Usa lo **stesso** meccanismo di invalidazione di 7bis.2, specificato in 7bis.5.

**Le chiavi AdaLN sono lette, non dedotte.** L'header del turbo upstream non
potato (`larryvrh/MiniMax-H3-Turbo-Lora`, `minimax_h3_turbo_4step.safetensors`,
779,8 MB) è stato riletto con una range request da 1 MB: 57.480 byte di header,
518 chiavi, 259 coppie, ranghi `{64: 208, 16: 51}`, nessun `.alpha`, nessun
prefisso. Verbatim:

```
blocks.N.adaln_proj.linear.lora_A.weight     BF16  [16, 2688]
blocks.N.adaln_proj.linear.lora_B.weight     BF16  [96768, 16]
final_layer.adaln_proj.linear.lora_A.weight  BF16  [16, 2688]
final_layer.adaln_proj.linear.lora_B.weight  BF16  [10752, 16]
```

Coincidono con quello che la regola di 7.2 deriva dal nome del bersaglio, quindi
la regola regge; ma `final_layer` non era enumerato, e con il fallimento fatale
della sezione 3 quel file sarebbe stato **interamente inutilizzabile** per 51
coppie su 259. È il prezzo che la sezione 3 mette a verbale, presentato dal primo
file reale che lo tocca.

Copertura dei bersagli nell'upstream, per il verbale: 200 coppie sulle quattro
proiezioni dei blocchi, 50 su `blocks.N.adaln_proj.linear`, 8 sulle quattro
proiezioni dei due blocchi del token refiner, 1 su `final_layer.adaln_proj.linear`.

Il LoRA di riferimento di questa sessione resta senza coppie AdaLN: il suo
convertitore le ha rimosse (sezione 7.2) perché puntavano a una variante
**pruned** di FL2VA il cui ingresso AdaLN è a 8 dimensioni, mentre la sorgente
era a 2688. `h3` carica la FL2VA **non pruned**, dove entrano esattamente. Ma
scaricare gli 780 MB **non serve**: vedi il corpus di test in sezione 8, dove il
caso AdaLN numerico si misura con una coppia sintetica alle forme reali contro
una `W` vera.

---

### 7bis.4 La forma del ramo sulla GPU

Il ramo si compone di **tre dispatch**, tutti su funzioni che esistono già:

```
hidden = A·x        h3_gpu_linear_bf16
delta  = B·hidden   h3_gpu_linear_bf16
y      = y + delta  h3_gpu_add_bf16
```

Nessun kernel Metal nuovo. Quattro decisioni lo fissano.

**La strength è fusa in `A`, non applicata a runtime.** Il fattore
`strength * alpha/rank` viene moltiplicato dentro la copia bf16 di `A` nel
momento in cui l'insieme attivo viene materializzato sulla GPU, **non** dentro
l'adapter parsato in cache (sezione 6ter: quello è indicizzato per
`path + size + mtime`). L'insieme attivo è sostituito per intero a ogni
generazione, quindi in memoria resta **una copia per adapter**, non una per
strength. Un cambio di strength rimaterializza `A`: se il parse è ancora in
cache è un ricalcolo in memoria, altrimenti sono 591 MiB a 4,28 GiB/s, cioè
**0,14 s**.

Costo numerico: un arrotondamento bf16 in più sui coefficienti di `A`, ordine
`1e-3` relativo, che vive solo dentro il delta. A `strength 100` il delta è il
4,7% dell'output, quindi il contributo al totale è circa `1e-4`, quindici volte
sotto il pavimento di rumore misurato di `1,65e-03` (sezione 8, T1).

**L'intermedio è bf16 e vive in un solo buffer condiviso.** `hidden` è
`righe x rank`, allocato al load e dimensionato su `max_rank` dell'insieme
attivo, riusato da ogni proiezione di ogni blocco. Il denoise oggi ha
`alloc=0.000GiB` sull'intero loop e deve restare così. Buffer separati per
adapter o per proiezione non comprano parallelismo, perché i dispatch sulla
stessa command queue sono comunque ordinati: comprano solo memoria.

**N adapter sono N rami sequenziali.** Nessuna `A` concatenata, nessuna `B` a
blocchi diagonali. È anche la scelta di `llama.cpp`
(`llama-graph.cpp:1521-1537`). H3 vale per entrambe le forme; vince la più
semplice, e la concatenata avrebbe uno stato derivato da ricostruire a ogni
`!lora add`.

**Il rank non viene imbottito a 256.** Motivo sotto.

#### Il costo misurato, e perché non si ottimizza

`h3_gpu_linear_bf16` instrada su MPS **solo** se
`rows >= 32 && input_dim >= 256 && output_dim >= 256` (`h3_gpu.m:2517`);
altrimenti cade sul kernel a piastrelle 16x16 scritto a mano
(`h3_gpu.m:2537`). Con rank 16..128 **entrambe** le metà del ramo mancano la
soglia: `A·x` ha output pari al rank, `B·hidden` ha ingresso pari al rank.

Misurato su M4 Pro a **4284 righe** (la geometria vera di un run
`448x576` / 56 frame), rank 64, mediane in ms:

| proiezione | BASE `W·x` | `A·x` | `B·hidden` | ADD |
|---|---:|---:|---:|---:|
| qkv | 138,03 | 3,11 | 12,28 | 4,74 |
| out | 45,85 | 4,08 | 3,21 | 1,31 |
| fc1 | 181,79 | 3,12 | 16,33 | 6,23 |
| fc2 | 91,48 | 8,08 | 3,20 | 1,31 |
| **totale** | **457,14** | **18,38** | **35,01** | **13,60** |

Il ramo costa il **14,7% del BASE**, circa il 7% del denoise, circa **54 s** sul
run da 56 frame di `AGENTS.md`. Il percorso è verificato per misura e non
assunto: `h3_gpu_stats` separa già `mps_linear_dispatches` da
`direct_dispatches`, e ogni riga del banco stampa quale dei due si è mosso.

Imbottire il rank a 256 con zeri fa entrare entrambe le metà in MPS e taglia il
ramo al **9,8% del BASE**, circa 18 s risparmiati. È bit-esatto (differenza
massima `0,000e+00` su tutte e quattro le proiezioni). **Resta fuori scope**
perché costa **4x la memoria degli adapter**, da 620 MB a 2,48 GB per il file di
riferimento, cioè **+1,86 GB residenti**: contro i circa 2,2 GiB di margine che
G4 misura a canvas massimo con `--preview`, quei 18 s si pagherebbero con la
capacità di caricare più di un adapter, che è il punto della feature.

Numeri agli atti per G6, se un giorno si riapre: il pareggio fra kernel naive e
MPS cade a **rank ~36**, sotto rank 32 imbottire fa danno, e il caso forte non è
il rank 64 ma il **rank 128**, dove non imbottire costa **3,4x**. L'ADD vale il
3% del BASE e fonderlo nell'epilogo della seconda GEMM varrebbe circa quanto il
padding.

Cautela sulla misura: ogni tempo è preso in un command buffer isolato con
`waitUntilCompleted`. Dentro il DiT vero questi dispatch stanno nello stesso
buffer del resto e la GPU può sovrapporli, quindi il **14,7% è un limite
superiore**. Codice del banco: `tests/bench_lora_gemm.c` su
`bench/lora-thin-gemm`.

---

### 7bis.5 Il meccanismo di invalidazione

**Non esiste un percorso di ricalcolo dedicato.** L'insieme attivo entra nella
chiave della cache del DiT preparato, `h3_prepared_key` (`h3.c:164`). Chiave
diversa dal DiT in cache e `h3.c:1502-1507` lo libera e lo ricarica: `refine_text`
(7bis.2) e il precompute AdaLN (7bis.3) si rifanno come parte di un load normale,
senza codice di invalidazione proprio.

Costo misurato di quella ricarica: **9,011 s**, 28,5 GiB letti (`logs/run02.log`,
30 step, 56 frame, `448x576`). Il precompute AdaLN è 24,2 di quei 28,5 GiB, cioè
50 blocchi da `96768 x 2688` in BF16, 496 MiB ciascuno.

**Alternativa rifiutata: il ricalcolo mirato.** Sarebbe più economico, perché
AdaLN è additivo: `time·(W + s·B·A)ᵀ = time·Wᵀ + s·(time·Aᵀ)·Bᵀ`. Si potrebbe
tenere la tabella base in memoria e sommarci il solo termine low-rank, senza
rileggere i 24,2 GiB, al prezzo di una seconda copia della tabella, circa 600 MB
a 30 step contro le ~2,2 GiB di margine al canvas massimo (sezione 10, G4).
È rifiutata **per T6, non per il costo**: rimuovere un adapter significherebbe
sottrarre un delta in floating point, e il terzo risultato non tornerebbe
bit-identico al primo. La ricarica riesegue il percorso del primo run, quindi T6
passa per costruzione.

**La chiave del conditioning non si tocca.** `h3_conditioning_key` (`h3.c:139`)
indicizza l'embedding del testo. Il LoRA agisce sul token refiner del DiT, non sul
text encoder Qwen, quindi i 9,596 s dell'encoder non si ripagano a ogni cambio.

**Identità di una voce dell'insieme**: una chiamata a `h3_key_file`
(`h3.c:126-136`) più la strength. Quell'helper aggiunge già percorso, `size` e
`mtime` in secondi e nanosecondi, ed è già usato per `--first-frame`,
`--last-frame` e i media di riferimento: è l'identità di file che la repo ha già,
non una nuova. `size` è ridondante come protezione dalle collisioni, con `mtime`
al nanosecondo su APFS; si tiene perché toglierlo vorrebbe dire scrivere una
seconda definizione divergente di "stesso file" a beneficio zero.

**La chiave si costruisce dall'insieme attivo già costruito**, non dalla lista
grezza in `h3_params`. Le voci a strength 0 sono già scartate in costruzione
(6ter), quindi una strength portata a 0 e poi indietro non conta come due cambi
e non serve un equivalente di `adapters_lora_are_same` di `llama.cpp`
(`src/llama-context.cpp:1293-1317`): la guardia è nel punto in cui la chiave si
costruisce, non in un confronto dedicato.

**Sensibile all'ordine, nessun sort.** I rami sono N somme in sequenza e la somma
in floating point non è associativa: `[A, B]` e `[B, A]` danno tabelle AdaLN
diverse negli ultimi bit. Un riordino costa una ricarica da 9 s, è raggiungibile
con `!lora remove` più `!lora add` ed è raro.

**Pigra, non immediata.** Il confronto avviene all'inizio di `h3_generate`, non
dentro `!lora add`, quindi un blocco di comandi `!lora` ricalcola una volta sola.
La validazione resta immediata: `h3_lora_preload` legge solo l'header safetensors
(6ter). Una ricarica immediata non risparmierebbe i 9 s, li anticiperebbe.

**`--reuse` e `--core-reuse`: niente da fare, verificato.** `core_reuse` è già in
`h3_prepared_key`, e il residuo cachato è `hidden - core_input`
(`h3_dit.c:2306-2310`), formato dopo che il delta è stato sommato dentro i
blocchi: una valutazione saltata (`h3_dit.c:2313-2315`) lo porta con sé.
`h3_dit_reset_run` (`h3.c:1470`) non richiede logica LoRA, perché un cambio
dell'insieme attivo è sempre un miss di cache.

---

## 8. Requisiti di test

Il repo ha già una suite estesa (`make test`, oltre venti binari fra cui
`h3_real_dit_block_test`, `h3_semantic_dit_test`, `h3_bf16_tests`). I test nuovi
si aggiungono lì, con lo stesso stile, non in un framework nuovo.

### 8bis. Il corpus

I test si dividono in due metà secondo il **soggetto**. Quando il soggetto è il
*contenuto* del file (nomi, forme, metadati, header rotto) il file si sintetizza.
Quando il soggetto sono i *numeri* (accordo con il riferimento float32, identità
byte a byte del video) servono checkpoint e LoRA reali. La linea non lascia zone
grigie: nessun test guarda un numero prodotto da un file inventato, e nessun test
aspetta un download per controllare una stringa.

**Metà sintetica: T3, T3b, T4, T4b.** Undici file, scritti dal binario di test in
una directory temporanea a ogni esecuzione e cancellati all'uscita. Nessun file
committato, nessuna voce nuova in `.gitignore`, nessun checksum, nessun percorso
da documentare. Safetensors è un `uint64` di lunghezza, un header JSON e byte
grezzi: generarlo in C è coerente con il "niente Python nel loop di test" che T1
già impone, e il generatore deve esistere comunque perché le fixture siano
verificabili, quindi committarne anche l'output sarebbe la stessa cosa scritta
due volte.

| # | Fixture | Preteso da |
|---:|---|---|
| 1 | convenzione A minima, con prefisso `diffusion_model.` | T3 |
| 2 | la stessa, senza prefisso | T3 |
| 3 | ibrida con `.alpha` e `alpha != rank` | T3, H6 |
| 4 | ranghi diversi dentro lo stesso file | T3 |
| 5 | convenzione B (`lora_up`/`lora_down`), da respingere | T3, 7.3 |
| 6 | `base_model` nei metadati non corrispondente, da avvisare | T3 |
| 7 | due o più coppie orfane | T4, H5 |
| 8 | coppia incoerente, rango di `A` diverso da quello di `B` | T4 |
| 9 | file troncato: header valido, dati incompleti | T4 |
| 10 | firma di conversione nei metadati | T4b |
| 11 | coppie con nome AdaLN | T4b |

T3b non aggiunge un dodicesimo file: è la fixture 1 scritta su un percorso che
contiene un `:`.

Poiché queste fixture non possono mancare, i test che le usano **non saltano
mai**. È questo che disinnesca la preoccupazione di 5.1: il conteggio AdaLN
sempre dichiarato e l'avviso di conversione sono verificati anche su una macchina
senza checkpoint, cioè proprio dove uno skip li avrebbe nascosti.

**Metà reale: T1, T1b, T2, T5, T6.** Due file, entrambi già presenti, nessun
download:

| File | Coppie | Rango | AdaLN | Copre |
|---|---:|---:|---|---|
| `minimax_h3_turbo_v4_step600_pruned_comfyui.safetensors` (591,6 MB) | 208 | 64 | no | T1, T1b, T2, T6 |
| `H3_Combat_V2.safetensors` (147,9 MB) | 208 | 16 | no | T1 a rango 16, T5 |

Il secondo chiude il "non misurato a rango 16" di T1 e dà a T5 due LoRA reali di
rango diverso invece dello stesso file usato due volte. È convenzione A, prefisso
`diffusion_model.`, nessun `.alpha`, metadati `ai-toolkit` con
`ss_base_model_version: minimax_h3`.

Il caso AdaLN numerico **non richiede il turbo upstream**: l'oracolo costruisce
`W + s·B@A` da una `W` reale e da una coppia qualunque della forma giusta, quindi
una coppia sintetica `A[16, 2688]` / `B[96768, 16]` (circa 3,2 MB) si misura
contro `blocks.N.adaln_proj.linear.weight` esattamente come le altre quattro
proiezioni. I nomi delle chiavi non sono dedotti: sono letti dall'header
upstream, vedi 7bis.3.

**Nessun LoRA entra nel repo.** GitHub rifiuta il push di qualunque file oltre i
100 MB e i due reali sono 591,6 e 147,9: passerebbero solo con Git LFS, che è una
dipendenza esterna in un progetto che non ne ha, che `antirez/h3.c` non usa, e
che farebbe pagare 740 MB a ogni clone per sempre, il doppio il giorno che un
file viene sostituito. Sono anche ridistribuzioni di terzi a licenza non
verificata.

**Assenza di un file reale: `skip:`, come il resto della suite.** `make test`
guarda ogni file con `test -f` e stampa `skip:` in tredici punti; la metà reale
segue quella convenzione e non ne inventa una seconda. I binari che pretendono i
62 GB stanno in un **target separato senza guardie**, sul modello di `make
parity` e `make real-parity`, che già falliscono duro quando i loro file non ci
sono. La divisione cade dove deve: `make test` resta verde su una macchina nuda,
quindi T7 resta verificabile senza checkpoint, e il target dedicato dichiara di
volerlo.

Il README guadagna una sezione di setup che elenca quali file installare e dove,
così il target dedicato ha una procedura invece di un messaggio d'errore.

**T1 — Equivalenza al peso fuso (l'oracolo).** L'oracolo è un **riferimento CPU
in float32 costruito dentro il binario di test in C**, non un artefatto generato
fuori: si prende un `W` reale dal checkpoint e una coppia `(A, B)` reale dal file
LoRA, si costruisce `W + s·B@A` in float32 sulla CPU, lo si applica a un `x`
fissato da seed, e lo si confronta con il ramo LoRA sulla GPU.
**Nessun Python nel loop di test.** **L'oracolo è per tensore, non per modello**:
bastano pochi secondi e nessuna copia da 62 GiB.

**Tolleranza: L2 relativo `<= 3e-3`, misurato a `strength 100`.** Un numero
solo, non uno per forma: le quattro proiezioni stanno entro l'1,8% l'una
dall'altra, splittarle non comprerebbe niente.

`1e-3` **non sopravvive**. Era scritto contro un oracolo a peso fuso in BF16, che
arrotonda `W + s·B@A` *prima* di applicarlo e porta quindi un errore proprio.
Contro il riferimento float32 il pavimento di errore misurato è **1,65e-03 a
1,68e-03**, già sopra la vecchia soglia: ereditarla farebbe fallire
un'implementazione corretta.

Perché `strength 100` e non 1: l'errore del ramo **non dipende dalla strength**
(piatto su 1,65e-03..1,68e-03 da `s = 1` a `s = 100`), mentre il peso del delta
cresce **linearmente** con essa. A `s = 1` il delta vale 1,7e-04 a 5,0e-04, cioè
*sotto* il pavimento di rumore, e un T1 a strength 1 passerebbe con il delta
cancellato. A `s = 100` il delta minimo misurato è **1,61e-02**, 5,4 volte la
soglia, quindi un delta assente, trasposto o mal scalato la sfonda. La strength
alta è un amplificatore di segnale per il test, non un valore d'uso.

La soglia si applica alla cifra **`bf16-out`**, la somma riarrotondata a BF16,
sempre la maggiore delle due stampate: così il numero regge qualunque esito abbia
la decisione aperta sul dtype di accumulazione. Peggiore misurato 2,374e-03,
margine 26%.

**Guardia sulla significatività**: T1 deve fallire anche quando il peso del delta
scende sotto `10x` la tolleranza. Senza, sostituire un file LoRA troppo debole
degrada il test in un no-op silenzioso.

Misurato con `h3_lora_oracle_test loras/turbo.safetensors MiniMax-H3 100 <block>`
sui blocchi 0, 10, 25, 44. `rel-max` resta non vincolato, peggiore misurato
5,49e-03. **Non misurato a rank 16**: il file locale è la turbo potata e tutte le
coppie sopravvissute sono a rank 64.

**T2 — Identità a strength zero.** `h3` con `--lora X --lora-strength 0` produce
un mp4 **byte-identico** a `h3` senza `--lora`, stesso seed e stessi parametri.
Copre H4.

**T1b — Oracolo di blocco intero.** Lo stesso confronto di T1, ma su **tutte e
quattro** le proiezioni di un blocco reale nello stesso passaggio. Motivo: un
oracolo per singolo tensore passa anche se una proiezione è trasposta rispetto a
un'altra, o se a una manca del tutto l'innesto. Solo il blocco intero lo vede.

**T3 — Parser.** File sintetici minuscoli in convenzione A, nella variante ibrida
con `.alpha` e `alpha != rank` verificando che la scala effettiva sia `alpha/rank`
(copre H6), con e senza il prefisso `diffusion_model.`, e con ranghi diversi
dentro lo stesso file. Un file in convenzione B deve essere respinto con l'errore
di 7.3. Un file il cui `base_model` nei metadati non corrisponde al checkpoint
caricato deve produrre un **avviso**, non un errore: il campo è informativo e
scritto dal convertitore.

**T3b — Parsing della riga di comando.** `--lora PATH[:STRENGTH]`: percorso nudo
(strength `1.0`), percorso con strength, ripetizione del flag, e **un percorso
che contiene esso stesso un `:`**. Lo split sull'ultimo `:` è la regola sottile
di G3 e va coperta da un test, non dalla lettura del codice.

**T4 — Rifiuto e segnalazione.** Tre casi, tutti fatali, con messaggi distinti:
un LoRA con coppie verso tensori che `h3` non applica deve fallire **elencandole
tutte**, non solo la prima (H5 e sezione 3 punto 3); una coppia internamente
incoerente, con i ranghi di `A` e `B` discordi, deve fallire; un file **troncato**,
cioè con un header valido e i dati incompleti, deve fallire con un errore che dice
che è troncato, non con un crash.

**T4b — Forma del report.** Su un file valido, la riga dei ranghi è l'istogramma
di sezione 5 punto 1 e il conteggio AdaLN c'è anche quando vale zero. Su un file
con firma di conversione, esce in più una riga `warning:` che contiene il valore
di `removed_pair_count` letto dall'header.

**T5 — Composizione.** Due LoRA attivi insieme danno lo stesso risultato di un
singolo LoRA i cui delta sono la somma dei due, entro la tolleranza di T1.

**T6 — Hot-swap.** In sessione: generare, aggiungere un LoRA, rigenerare con lo
stesso seed, rimuoverlo, rigenerare. Il terzo risultato è identico al primo
**e il secondo è diverso dal primo**. Due asserzioni, non una: con la sola prima,
T6 è verde anche quando il delta non viene mai applicato, perché tre run del
modello base sono identici fra loro. Un test che verifica solo "tornando indietro
ritrovo lo stato di prima" non distingue lo scambio funzionante dal nulla che
accade.

Il secondo run gira a una strength di produzione, **non** alla 100 di T1: su una
traiettoria di denoise intera una perturbazione da 1e-04 nel blocco 0 diverge, e
la disuguaglianza a livello di byte è tutto il controllo che serve, senza metrica
di distanza. L'accettazione qualitativa sul video resta separata e a carico del
proprietario.

**T7 — Non regressione.** `make test` passa interamente, invariato.

---

## 9. Criterio di completamento

Le quattro barre. Se una sola non passa, si corregge e si **rivalida tutto
l'insieme**, non solo il pezzo toccato.

1. **Struttura file** — `h3_lora.c`/`h3_lora.h` esistono, nessuna dipendenza
   esterna aggiunta, `make -j8` pulito senza warning nuovi.
2. **Architettura** — nessuna scrittura sui tensori base (H1, H2), verificabile
   leggendo gli innesti; adapter allocati e liberati indipendentemente dal modello.
3. **Comportamento** — T1..T7 verdi.
4. **Ambiente E2E** — un video reale generato con il turbo LoRA attivo a
   `--steps 6 --reuse 1 --ssd-streaming`, confrontato con lo stesso prompt e
   seed senza LoRA. Il confronto è **osservativo**: serve a dimostrare che la
   pipeline regge end to end, non a giudicare la qualità del LoRA.

---

## 10. Vincoli aperti

| # | Gap | Stato |
|---|---|---|
| G1 | `adaln_proj`: **supportato**, e sono **due** bersagli, non uno: `blocks.N.adaln_proj.linear.weight` più `final_layer.adaln_proj.linear.weight` (sezione 7.1), che è la coppia numero 51. Non è una quinta proiezione per blocco ma un precalcolo una tantum (sezione 7bis.3), invalidato da un cambio di LoRA con lo stesso meccanismo del refiner. Il LoRA di riferimento ha le 51 coppie rimosse, ma il caso **è** verificabile: coppia sintetica alle forme reali contro una `W` vera, con le chiavi lette dall'header upstream (sezione 8bis). | **chiuso, supportato** |
| G2 | `--lora` funziona **sia** in one-shot (`h3 -p ...`) **sia** in sessione interattiva. Motivo: T2 confronta due invocazioni di `h3` a parità di seed, che *è* modalità one-shot — il piano di test la richiedeva già. | **chiuso** |
| G3 | Sintassi fissata: **`--lora PATH[:STRENGTH]`**, ripetibile, strength opzionale con default `1.0`, parsata **splittando sull'ultimo `:`**. Superficie interattiva: `!lora add PATH [STRENGTH]`, `!lora set PATH STRENGTH`, `!lora remove PATH`, `!lora` nudo elenca. Motivo: LoRA e strength non sono separabili da un errore dell'utente, a differenza di flag appaiati dove una strength omessa sposta tutti gli argomenti successivi. | **chiuso** |
| G4 | Tetto di memoria. **Chiuso**: nessun tetto statico sugli adapter. Il tetto è sul **processo intero**, misurato e mai previsto, e al tetto h3 **si ferma**. Vale `min(recommendedMaxWorkingSetSize - 4 GiB, libera di sistema + footprint attuale)`, verificato a due cancelli, ed è **sempre attivo** anche senza LoRA: allargamento dichiarato da "tetto degli adapter" a "guardrail di memoria di h3". Dettaglio sotto. | **chiuso** |
| G5 | I percorsi residente e int8 restano espressamente **fuori scope** (decisione di Fase 3). Da riaprire solo dopo che lo streaming funziona. | **chiuso, escluso** |
| G6 | Budget prestazionale. Deciso in Fase 3: **non è un vincolo ora**, prima correttezza poi ottimizzazione. Il numero **non** passa da `--profile` (requisito ritirato, sezione 5 punto 3): si misura fuori banda con run A/B quando serve. | **chiuso, rimandato** |
| G7 | Il turbo LoRA specifico funzioni o no è **irrilevante** per l'accettazione: il progetto consegna il meccanismo, non la qualità di un file di terze parti. | **chiuso, fuori scope** |

### G4 — il guardrail di memoria

**Nessun tetto statico sugli adapter.** Un numero fisso sui byte degli adapter
non si applica alla condizione in esecuzione: la leva che muove davvero il picco
è il canvas, e il picco di h3 **non è noto prima di girare** (il termine
dipendente dalla risoluzione sta fuori da `h3_gpu_stats`, che riporta 2,58 GiB
sia a `448x576` sia a `768x1344`). Quindi non si prevede, si misura.

Il tetto è sul processo intero:

```
tetto_statico  = recommendedMaxWorkingSetSize - 4 GiB      /* 32 GiB qui */
tetto_dinamico = libera_di_sistema + footprint_attuale
tetto           = min(tetto_statico, tetto_dinamico)
```

`recommendedMaxWorkingSetSize` vale 36 GiB su questa macchina, la stessa cifra
che LM Studio mostra come "VRAM 36.00 GB". La **riserva di 4 GiB è il solo numero
fisso di tutto il design**, giustificata da cosa deve restare in piedi mentre h3
tiene 22 GB (sistema, WindowServer, compressore) e non da una stima del nostro
consumo: **nessun flag, nessun preset, nessuna sovrascrittura**.

`libera_di_sistema` è `free_count + inactive_count + purgeable_count` da
`host_statistics64(HOST_VM_INFO64)`, in `<mach/mach.h>`, zero dipendenze. Le
pagine che un altro processo tiene `active` o `wired` (un server LLM, un altro
modello video) non contano come libere: il termine statico è cieco a quel caso,
il dinamico no. Il messaggio di stop nomina **quale dei due termini ha morso**,
altrimenti l'utente non sa se chiudere l'altro processo o abbassare il canvas.

**Cancello 1, pre-flight, prima di leggere qualunque byte.** Confronta il tetto
con la somma dei soli termini noti **esattamente** e indipendenti dal canvas: i
due slot BF16 di pesi (1,5 GB, aritmetica sotto), il decoder VAE tiled
(9,55 GiB misurati, si muove dell'1,7% per 4x i pixel) e i byte residenti degli
adapter, noti dagli header safetensors al preload. Con `--preview` o DiT cachato
gli ultimi due sono concorrenti, quindi ~11 GB più adapter; altrimenti gli stage
sono serializzati (`h3.c:1591`) e il massimo è il decoder.

Resta una **condizione necessaria e non sufficiente**, perché non modella il
termine dipendente dal canvas del denoise: quello lo vede solo il cancello 2. Ma
il blocco degli encoder, che prima era la lacuna dichiarata di questo calcolo,
ora è **misurato e non lo cambia**. Il picco di processo attraverso tokenizer,
video VAE encoder, Qwen vision e Qwen text encoder sta fra **4,7 e 6,0 GB** su
sei run, cioè circa il 40% sotto i 9,55 GiB del decoder, che resta il massimo
degli stage.

Né il canvas né la lunghezza del prompt lo muovono in modo misurabile: la
dispersione fra due run della **stessa** configurazione (1,1-1,3 GB) copre per
intero l'intervallo fra le quattro configurazioni provate (`448x576` e
`768x1344` per 1640 e 6554 caratteri di prompt). La parte deterministica è il
contatore interno di h3 per il solo Qwen text encoder, che si ripete a quattro
cifre fra run identici e sale appena da **2,846 a 3,204 GiB** dal caso più
piccolo al più grande: il resto è slack dell'allocatore e del driver, non un
termine da modellare.

**Cancello 2, dopo la prima valutazione del denoiser.** Lì il footprint è a
regime per misura (`alloc=0.000GiB` su tutto il loop di denoise). Si campiona
`phys_footprint` con `task_info(mach_task_self(), TASK_VM_INFO, ...)`, lo stesso
numero che legge `footprint -p` e il solo che vede il termine dipendente dalla
risoluzione. Se il tetto è superato, si ferma. Costa una valutazione, ~45 s su un
run da 13 minuti, e sostituisce una previsione con un fatto. È il cancello 2 che
rende inutile un budget statico: il limite segue i tensori davvero in uso.

**Al tetto si ferma, mai un set parziale.** Il set attivo è dichiarativo e
sostituito in blocco a ogni generazione (sezione 6ter), quindi "rifiuta
l'N-esimo" non è definito, e un adapter scartato in silenzio è ciò che **H5**
vieta. In one-shot `h3_generate` fallisce al cancello 1, prima della lettura del
checkpoint. In sessione il cancello rifiuta **la generazione**, non l'`!lora add`,
la cui validità resta governata da G3 e dalla sezione 6ter.

**Allocazione eager**, al preload e all'`!lora add`: è l'unico modo in cui
"blocca subito" ha un significato. Limite noto: al momento dell'add il termine
del canvas non è allocato, quindi un add che passa può comunque portare la
generazione successiva in swap. È esattamente il motivo per cui esiste il
cancello 2.

**Cosa conta nel budget**: i byte **residenti** di `A` e `B` delle sole coppie
accoppiate a un bersaglio, nel dtype residente. Le coppie riportate e saltate
(sezione 3.3) non sono residenti. Le entry a strength 0 sono già scartate alla
costruzione del set (sezione 6ter). Un file float32 conta il doppio: il budget
misura byte residenti, non byte su disco.

**I cancelli sono sempre attivi**, anche con set vuoto: misurano il processo
intero, quindi un branch in meno e protezione anche nei run senza adapter, che
sono quelli che oggi vanno in swap. Questo è un **allargamento dichiarato** di
G4, non un effetto collaterale.

**Lo spill su SSD resta un riempimento successivo**, con l'innesco ormai fissato:
il cancello 2, mai una soglia inventata. L'indirezione `source` nella struct
dell'adapter resta come predisposizione. Per memoria: costerebbe ~7-10 GiB di
letture in più sui 575 GiB già letti, il ~2%, contro un `unhidden wait 0.006s`
misurato. Poco, ma un abort è verificabile in un test e uno spill sotto pressione
di memoria no.

#### L'aritmetica di supporto

Da `h3_dit.h:21-23`, lo streaming trattiene solo le norme di blocco più **due**
slot BF16 alternati di matrici. Dalle forme della sezione 7.1, uno slot vale
`21504·5376 + 5376·7168 + 28672·5376 + 5376·14336 = 385,4M` elementi
≈ **771 MB** in BF16, quindi due slot ≈ **1,5 GB** residenti.

Dal lato adapter, un rank-64 a copertura piena vale
`(64·5376 + 21504·64) + (64·7168 + 5376·64) + (64·5376 + 28672·64) + (64·14336 + 5376·64)`
= 5,96M elementi per blocco, cioè 11,4 MiB per blocco su 52 blocchi = **591,7 MiB**,
più ~57 MB per le 51 coppie AdaLN a rango 16. Verificato contro il file reale:
`loras/turbo.safetensors` misura **591,6 MiB**. I LoRA per H3 sono quindi 3-4x
quelli SDXL/Wan, perché hidden è 5376, e **4-5 adapter insieme sono 2,4-3,0 GB
residenti**.

Conseguenza da tenere presente: con tetto a 32 GiB e la configurazione peggiore
raggiungibile (canvas massimo con `--preview` o DiT cachato, picco ~32 GB),
l'headroom per gli adapter è **~2,2 GiB**, cioè tre adapter rank-64 pieni e non
cinque. Quella combinazione **è attesa far scattare il cancello 2**, e fermarsi
lì è il comportamento voluto, non un difetto. Al canvas della ricetta l'headroom
è ~20 GiB e nulla morde.

---

## 11. Fonti

- **Scope, le quattro decisioni, il caso d'uso** — sessione con NeoKree del
  2026-09-01/02. Decisioni di Fase 3: solo `--ssd-streaming`; più LoRA con
  hot-swap in sessione; rilevamento automatico del formato (**superata**, vedi
  7.3: la convenzione B è rilevata e rifiutata); prestazioni non vincolanti in
  questo giro.
- **Punti di innesto e vincoli di percorso** — lettura diretta di `h3_dit.c`
  (righe 502, 517-522, 615-618, 804-880, 819-821, 869, 1556, 1633-1639, 1667,
  1670), `h3_dit_schedule.c` (267), `h3_gpu.m` (364-373, 3780, 3805-3810),
  `h3_gpu.h` (98-99), `h3_cli.c` (485, 588), `main.c` (270, 357), `h3_dit.h`
  (21-23, 31, 59), `h3.c` (553).
- **Comportamento di `--ssd-streaming`** — `README.md` righe 131-158 e 547-556,
  più il profilo misurato in sessione: 574,937 GiB letti a 4,280 GiB/s con
  `unhidden wait 0.006s` su un render da 56 frame.
- **Schema e metadati del LoRA** — header di
  `minimax_h3_turbo_v4_step600_pruned_comfyui.safetensors`, letto in sessione.
- **Forme dei tensori e copertura in byte** — header dei 13 shard di
  `FL2VA/transformer`, letti in sessione.
- **Arte nota** — lettura di `ggml-org/llama.cpp` al commit
  `de8656bd94f1163188125542534e4bcbc9f9fb1f` (`src/llama-adapter.cpp`,
  `src/llama-graph.cpp`, `src/llama-context.cpp`, `include/llama.h`) e di
  `leejet/stable-diffusion.cpp` (`src/model/adapter/lora.hpp`). Note complete in
  `docs/research/llama-cpp-adapters.md`.
