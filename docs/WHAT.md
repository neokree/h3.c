# WHAT.md — LoRA dinamici in h3-metal

## 0. Header

**Fonti consultate**

| Fonte | Cosa ne viene |
|---|---|
| Sessione di lavoro del 2026-09-01/02 con NeoKree | scope, le quattro decisioni di Fase 3, il caso d'uso turbo LoRA |
| Codice `h3.c` allo stato del commit `8974cc0` | punti di innesto, vincoli di percorso, infrastruttura di test |
| `README.md` del repo | comportamento documentato di `--ssd-streaming` e dello schedule |
| `minimax_h3_turbo_v4_step600_pruned_comfyui.safetensors` | schema del file LoRA, verbatim in sezione 7 |
| Header del turbo LoRA upstream non pruned (letto in sessione) | prefisso opzionale, ranghi non uniformi, coppie AdaLN presenti |
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
   si ferma, il messaggio d'errore le elenca **tutte**, e nessuna generazione
   parte. Sono fatali anche un file illeggibile o non parsabile e una coppia
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

---

## 5. Reportistica

Non c'è dashboard né export. Gli output secondari richiesti sono tre, tutti
testuali:

1. **All'attivazione di un LoRA**: percorso, **distribuzione dei ranghi** (non un
   rango unico, vedi 7.2), numero di coppie applicate, **numero di coppie AdaLN**,
   memoria occupata. Non esiste una sezione "coppie saltate": una coppia
   inapplicabile è fatale (sezione 3, punto 3) e il suo elenco esce dal messaggio
   d'errore, non dal report.

   **Forma della distribuzione dei ranghi**: istogramma compatto su una riga,
   ordinato per conteggio decrescente, `ranks: 64 x208, 16 x51`. Non un intervallo
   (`16-64` nasconde la bimodalità, che è il segnale utile: 16 è l'AdaLN, 64 il
   backbone) e non un raggruppamento per tipo di bersaglio, che inventerebbe una
   tassonomia da mantenere a ogni bersaglio nuovo.

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
   `typedef int (*h3_report_callback)(const char *line, void *opaque)`, stessa
   forma di `h3_progress_callback`. La libreria non stampa da sé: chi integra
   `libh3.a` decide dove finisce il testo, e H5 esige che si veda.
5. **Chiave di cache: path più dimensione più mtime** (`stat`). Un file
   sovrascritto allo stesso path viene ricaricato invece di restare quello
   vecchio in memoria senza dirlo.
6. **Un caso fatale ferma la generazione.** `h3_generate` fallisce e il
   messaggio esce da `h3_last_error`. Non si consegna mai un video privo del
   LoRA richiesto: a tredici minuti per run, non c'è modo di accorgersene
   guardandolo. Vale solo per i due casi fatali di sezione 3 punto 3.
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

Copertura: **208 tensori su 535, pari al 60,5% del DiT in byte** (37,3 GiB su 61,7).

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

`adaln_proj` **non** è quindi una quinta proiezione per blocco accanto alle
quattro della sezione 7.1: strutturalmente è lo stesso problema del refiner, un
precalcolo una tantum invalidato da un cambio di LoRA, **non** una proiezione
per-step in streaming. Usa lo **stesso** meccanismo di invalidazione di 7bis.2.

Limite noto: non è testabile con il LoRA di riferimento di questa sessione, le
cui 51 coppie AdaLN sono state rimosse dal suo convertitore (sezione 7.2).

Quella rimozione, però, non riguarda `h3`. Il convertitore ha potato le coppie
perché puntavano a una variante **pruned** di FL2VA il cui ingresso AdaLN è a 8
dimensioni, mentre la sorgente era a 2688. `h3` carica la FL2VA **non pruned**,
dove quelle coppie entrano esattamente. Sono quindi recuperabili scaricando il
turbo LoRA upstream (circa 780 MB), che le porta a rango 16. Verificato a livello
di byte sugli header dei due file.

---

## 8. Requisiti di test

Il repo ha già una suite estesa (`make test`, oltre venti binari fra cui
`h3_real_dit_block_test`, `h3_semantic_dit_test`, `h3_bf16_tests`). I test nuovi
si aggiungono lì, con lo stesso stile, non in un framework nuovo.

**T1 — Equivalenza al peso fuso (l'oracolo).** L'oracolo è un **riferimento CPU
in float32 costruito dentro il binario di test in C**, non un artefatto generato
fuori: si prende un `W` reale dal checkpoint e una coppia `(A, B)` reale dal file
LoRA, si costruisce `W + s·B@A` in float32 sulla CPU, lo si applica a un `x`
fissato da seed, e lo si confronta con il ramo LoRA sulla GPU.
**Nessun Python nel loop di test.** **L'oracolo è per tensore, non per modello**:
bastano pochi secondi e nessuna copia da 62 GiB.

Tolleranza: l'attuale `<= 1e-3` di L2 relativo era scritta contro un oracolo a
**peso fuso in BF16** e **va ri-derivata** contro il riferimento float32. Motivo:
un peso fuso in BF16 arrotonda `W + s·B@A` *prima* di applicarlo, e porta quindi
un errore proprio che il riferimento float32 non ha. La soglia nuova si misura,
non si eredita.

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
stesso seed, rimuoverlo, rigenerare. Il terzo risultato è identico al primo.

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
| G1 | `adaln_proj`: **supportato**. Non è una quinta proiezione per blocco ma un precalcolo una tantum (sezione 7bis.3), invalidato da un cambio di LoRA con lo stesso meccanismo del refiner. Non verificabile con il LoRA di riferimento, che ha le 51 coppie AdaLN rimosse. | **chiuso, supportato** |
| G2 | `--lora` funziona **sia** in one-shot (`h3 -p ...`) **sia** in sessione interattiva. Motivo: T2 confronta due invocazioni di `h3` a parità di seed, che *è* modalità one-shot — il piano di test la richiedeva già. | **chiuso** |
| G3 | Sintassi fissata: **`--lora PATH[:STRENGTH]`**, ripetibile, strength opzionale con default `1.0`, parsata **splittando sull'ultimo `:`**. Superficie interattiva: `!lora add PATH [STRENGTH]`, `!lora set PATH STRENGTH`, `!lora remove PATH`, `!lora` nudo elenca. Motivo: LoRA e strength non sono separabili da un errore dell'utente, a differenza di flag appaiati dove una strength omessa sposta tutti gli argomenti successivi. | **chiuso** |
| G4 | Tetto di memoria per gli adapter residenti. **Deciso**: adapter residenti in memoria GPU, ma la struct dell'adapter porta un'indirezione `source` fin dal primo giorno, così che lo spill su SSD sia un riempimento successivo e non un refactor. Il **numero** del tetto è deliberatamente non fissato: aspetta una misura di picco RSS sotto carico reale di generazione (vedi nota sotto). | **aperto, bloccato su misura** |
| G5 | I percorsi residente e int8 restano espressamente **fuori scope** (decisione di Fase 3). Da riaprire solo dopo che lo streaming funziona. | **chiuso, escluso** |
| G6 | Budget prestazionale. Deciso in Fase 3: **non è un vincolo ora**, prima correttezza poi ottimizzazione. Il numero **non** passa da `--profile` (requisito ritirato, sezione 5 punto 3): si misura fuori banda con run A/B quando serve. | **chiuso, rimandato** |
| G7 | Il turbo LoRA specifico funzioni o no è **irrilevante** per l'accettazione: il progetto consegna il meccanismo, non la qualità di un file di terze parti. | **chiuso, fuori scope** |

### G4 — l'aritmetica già assestata

Quello che **è** noto. Da `h3_dit.h:21-23`, lo streaming trattiene solo le norme
di blocco più **due** slot BF16 alternati di matrici. Dalle forme della sezione
7.1, uno slot vale
`21504·5376 + 5376·7168 + 28672·5376 + 5376·14336 = 385,4M` elementi
≈ **771 MB** in BF16, quindi due slot ≈ **1,5 GB** residenti. La macchina ha
48 GiB.

Quello che **non** è noto, ed è ciò su cui G4 è bloccato: il picco di occupazione
altrove nella pipeline. Anche la directory `text_encoder` è 62 GB su disco e
`video_vae` è 9,7 GB; **se vengano liberati prima del loop del denoiser non è
stato misurato**. Il tetto si fissa dopo quella misura, non prima.

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
