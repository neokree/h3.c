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
| `gh api` su `ggml-org/llama.cpp` e `leejet/stable-diffusion.cpp` (2026-09-02) | scelta della BAR |
| Lettura integrale di `llama-adapter.cpp` / `llama-graph.cpp` / `llama-context.cpp` (`docs/research/llama-cpp-adapters.md`, branch `research/llama-cpp-adapters`) | politica di fallimento, rango per coppia, aggiunte al piano di test |

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
   o la cui forma non entra in un bersaglio supportato, viene segnalata e
   saltata**: il caricamento prosegue. Sono fatali due soli casi: un file
   illeggibile o non parsabile, e una coppia internamente incoerente, cioè con il
   rango di `A` e quello di `B` discordi.

   Questa regola **sostituisce** la formulazione precedente di questo punto
   ("una incompatibilità è un errore, non un avviso"), che era la politica della
   BAR: `llama.cpp` aborta sulla prima coppia orfana (`llama-adapter.cpp:331`).
   La divergenza è deliberata. Un LoRA reale contiene coppie destinate a varianti
   del modello che `h3` non carica, e abortire renderebbe inutilizzabile un file
   per il resto valido. Il prezzo di questa scelta è la regola **H5**, che rende
   ogni salto visibile.
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
che puntano a tensori che `h3` non applica, l'utente deve vederlo elencato.
Un turbo LoRA a cui manca metà degli adapter deve *dirlo*, non degradare in
silenzio — è esattamente il modo in cui il file di prova di questa sessione
sarebbe passato inosservato.

Il **salto** è consentito (sezione 3, punto 3), il **silenzio** no. Anche qui la
BAR è dietro: `llama.cpp` salta senza dire niente le coppie `_norm.weight`, sotto
un TODO (`llama-adapter.cpp:287-290`). H5 vieta esattamente questo.

**H6 — La scala dichiarata dal file vince.** Nelle convenzioni che portano
`alpha`, il fattore effettivo è `alpha/rank` e va letto dal file, non assunto.
La `strength` dell'utente moltiplica quel fattore, non lo sostituisce.

---

## 5. Reportistica

Non c'è dashboard né export. Gli output secondari richiesti sono tre, tutti
testuali:

1. **All'attivazione di un LoRA**: percorso, **distribuzione dei ranghi** (non un
   rango unico, vedi 7.2), numero di coppie applicate, numero di coppie saltate
   con il motivo (regola H5), **numero di coppie AdaLN**, memoria occupata.

   Il conteggio delle coppie AdaLN va dichiarato **sempre**, anche quando è zero.
   Zero coppie AdaLN è normale in un LoRA di stile e patologico in un LoRA turbo,
   e solo l'utente sa quale dei due sta caricando. In più, se l'header del file
   porta una firma di conversione (`partial_conversion`, `removed_pair_count`,
   `adaln_keys_removed`), va emesso un avviso esplicito che la cita: quel file è
   stato potato da un convertitore e il numero di coppie mancanti è scritto lì.
2. **`!status`** deve elencare i LoRA attivi con la rispettiva strength.
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

## 6bis. Struttura repo e riferimento

### Struttura

Il codice nuovo vive accanto all'esistente, senza cartelle nuove:
`h3_lora.c` / `h3_lora.h` per il caricamento e la rappresentazione, innesti
puntuali in `h3_dit.c`, opzione in `main.c`, comando in `h3_cli.c`.

Nessuno script di supporto in Python: la directory `tools/` è stata eliminata e
non è mai stata committata. L'oracolo di sezione 8 è **dentro il binario di test
in C**, non fuori.

### BAR — `ggml-org/llama.cpp`

Verificato via `gh api repos/ggml-org/llama.cpp` il 2026-09-02:
`archived: false`, 126.694 star, ultimo push `2026-09-02T00:13:56Z`, linguaggio C++
con API pubblica in C.

**Perché questo e non `leejet/stable-diffusion.cpp`.** Sono stati esaminati
entrambi. Il marcatore che decide non è il nome della cartella ma **come il
delta raggiunge il peso**:

| Repo | File | Marcatore | Esito |
|---|---|---|---|
| `leejet/stable-diffusion.cpp` | `src/model/adapter/lora.hpp` | `ggml_add_inplace(compute_ctx, model_tensor, diff)` (riga 937) | **scartato**: somma nel tensore base, incompatibile con H2 |
| `ggml-org/llama.cpp` | `src/llama-adapter.cpp`, `include/llama.h` | `llama_set_adapters_lora(ctx, adapters, scales, n_adapters)` (riga 713), `llama_adapter_lora_init` (riga 680), `llama_adapter_lora_free` (riga 704) | **scelto**: adapter fuori dai pesi, N simultanei, scala per adapter, liberabili a caldo |

`stable-diffusion.cpp` è più vicino per dominio (diffusione, non LLM) ma la sua
architettura è precisamente quella che questo progetto vuole abbandonare.
`llama.cpp` è più lontano per dominio e più vicino per forma, ed è la forma che
conta: la sua API a tre funzioni è il modello diretto di ciò che serve qui.

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
internamente incoerente e il caso è fatale (sezione 3, punto 3). La BAR fa la
stessa cosa, leggendo il rango per coppia da `b->ne[0]` (`llama-adapter.cpp`).

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

Quel `warning` è il motivo per cui la regola **H5** esiste.

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

**T4 — Rifiuto e segnalazione.** Tre casi distinti, con esiti distinti:
un LoRA con coppie verso tensori che `h3` non applica deve **elencarle e
proseguire** (H5 e sezione 3 punto 3); una coppia internamente incoerente, con i
ranghi di `A` e `B` discordi, deve **fallire**; un file **troncato**, cioè con un
header valido e i dati incompleti, deve fallire con un errore che dice che è
troncato, non con un crash.

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
- **BAR** — `gh api repos/ggml-org/llama.cpp` e
  `gh api repos/leejet/stable-diffusion.cpp`, entrambi interrogati il 2026-09-02,
  più il contenuto di `include/llama.h` e `src/model/adapter/lora.hpp`.
