# Teknisk rapport: Segmentering for innsamling av bakkesannheter

_Oppdatert metodebeskrivelse for 2026_

## 1. Innledning

Formålet med arbeidet var å produsere segmenter som kan brukes til innsamling av bakkesannheter for bruk i produksjon av kart med fjernmåling og maskinlæring. Segmenteringen ble gjennomført i eCognition basert på en samlet rasterstack med CIR-ortofoto, avledede høydelag fra nasjonal detaljert høydemodell (NHM), NDVI og et vektordatasett fra AR5 som utgangspunkt for første segmentering.

## 2. Datagrunnlag

Datagrunnlaget i 2026 bestod av CIR-ortofoto, høydedata fra NHM og AR5. CIR-rasteret ble brukt som referanse for grid, koordinatsystem, oppløsning og utstrekning. Høydedata ble hentet som terrengmodell (DTM) og overflatemodell (DOM), og bare DTM- og DOM-fliser som overlappet CIR-utstrekningen ble brukt i videre prosessering.

AR5 ble brukt som vektordatasett i eCognition. Rulesetet importerer AR5 i setup-fasen, og segmenteringsrulesetet bruker AR5 som temalag ved vektorbasert segmentering.
Python-skriptet cir_topo_stack_gpu.py produserte én standard GeoTIFF med 10 bånd. I skriptet er beregningsbackend konfigurerbar som GPU/CuPy eller CPU/SciPy; den vedlagte konfigurasjonen er satt til CPU. Output er konfigurert som LZW-komprimert uint8, der gyldige verdier skaleres til 0-254 og verdien 255 er reservert som NoData.

**Merk:** Python-skriptets dokumenterte båndrekkefølge angir NIR, Red og Green som de tre første båndene, mens setup-rulesetet i eCognition gir aliasene 1_red, 2_green og 3_nir til de tre første importerte lagene. Dette bør kontrolleres ved import, slik at spektralbåndene får riktig alias før segmentering kjøres.

## 3. Behandling av høydedata og avledning av kartlag

For 2026 ble dataklargjøringen samlet i et Python-skript som bygger en komplett CIR/topografi-stack for hvert område. Skriptet leser først CIR-rasteret og bruker dette som referansegrid. Deretter skannes mapper med DTM- og DOM-fliser, og bare fliser med geografisk overlapp mot CIR-rasteret tas med. Overlappende DTM- og DOM-fliser mosaikkeres virtuelt med GDAL VRT og resamples til CIR-gridet med bilineær resampling.

Rasterprosesseringen skjer blokkvis for å håndtere store datasett effektivt. Hver prosesseringsblokk utvides med en halo slik at Gaussian-filtrering, TPI-beregninger og gradientbasert helning ikke får kantartefakter ved blokkgrensene. Skriptet bruker en blokkstørrelse på 8192 piksler, beregner nødvendige halo-størrelser fra filter- og vindusstørrelser, og skriver resultatet som standard GeoTIFF.

### 3.1 Canopy Height Model (CHM)

CHM ble beregnet ved å trekke DTM fra DOM. Negative høydeverdier ble satt til null. CHM inngår som eget bånd i 2026-stacken.

### 3.2 CHM_Gaus5

CHM_Gaus5 ble beregnet ved å filtrere CHM med et Gaussian-filter med sigma = 5 m. Siden filteret beregnes på CIR-gridet, omregnes sigma fra meter til piksler basert på rasterets faktiske oppløsning.

### 3.3 Topographic Position Index (TPI20, TPI50 og TPI70)

I 2026 ble det produsert tre TPI-lag. TPI beregnes fra DTM ved å sammenligne pikselhøyden med lokal middelhøyde i kvadratiske nabolag på 20, 50 og 70 piksler. Resultatene inngår som båndene TPI20, TPI50 og TPI70.

### 3.4 Slope_Gaus5

Helning ble beregnet fra DTM ved hjelp av gradienter i x- og y-retning og uttrykt i grader. Helningslaget ble deretter glattet med Gaussian-filter med sigma = 5 m.

### 3.5 NDVI

NDVI beregnes i Python-skriptet som (NIR - Red) / (NIR + Red). I 2026 inngår NDVI som et ferdig rasterbånd i inputstacken.

## 4. Oppsett i eCognition

eCognition-oppsettet er i 2026 delt i to ruleset-filer:

1. segment2026_ruleset_setup.dcp
2. segment2026_ruleset_segment.dcp

### Setup-ruleset

Gjør følgende:

- Import AR5.
- Create/modify project. (importerer kartlagene. Må sørge for at paths er riktige)
- Gi rasterlagene alias: 1_red, 2_green, 3_nir, 4_TPI20, 5_TPI50, 6_TPI70, 7_CHM_gaus5, 8_CHM, 9_NDVI og 10_Slope_Gaus5.
- Slette 'New Level'.

### Segmentering-ruleset

Lager segment A og B:

**Segment A:**

1. Vectorbased segmentation: creating 'L0' using AR5
2. Multi-threshold segmentation: creating 'L1' using TPI50:
   - Threshold: 1.2
   - Minimum object size: 750
3. Multi-threshold segmentation: creating 'L2' using CHM_gaus5:
   - Threshold: 0.4
   - Minimum object size: 1000
4. Multi-resolution segmentation: creating 'L3':
   - Scale: 25
   - Shape: 0.4
   - compactness: 0.5
   - Weights per layer:
     - Slope gaussian 5m: 2
     - Red band: 1
     - Green band: 2
     - NIR band: 3
     - TPI 50m: 2
     - CHM: 5
     - NDVI: 1

**Segment B:**

1. Multi-resolution segmentation: creating 'L3':
   - Scale: 8
   - Shape: 0.2
   - compactness: 0.5

- Weights per layer:
  - Slope gaussian 5m: 5
  - Red band: 1
  - Green band: 1
  - NIR band: 1
  - TPI 20m: 5
  - TPI 70m: 5
  - CHM: 5
  - NDVI: 1

## 6. Resultater

Metoden for 2026 gir to separate segmenteringsresultater: Segment A og Segment B. Segment A og Segment B segmenteres separat.

## 7. Konklusjon

2026-metoden bruker en samlet 10-bånds rasterstack og AR5 som vektordatasett i eCognition. Segmenteringen er delt i to separate løp: Segment A og Segment B.

## 8. Vedlegg

- **Vedlegg 1:** [script/cir_topo_stack_gpu.py](../scripts/cir_topo_stack_gpu.py)

  Python-skript brukt til å bygge inputstacken for 2026. Skriptet leser CIR, DTM og DOM, finner overlappende høydedatafliser, bygger VRT-mosaikker, resampler høydedata til CIR-gridet og beregner NIR, Red, Green, TPI20, TPI50, TPI70, CHM_Gaus5, CHM, NDVI og Slope_Gaus5. Skriptet støtter både GPU/CuPy og CPU/SciPy og skriver en standard GeoTIFF med 10 bånd.

- **Vedlegg 2:** [segment2026_ruleset_setup.dcp](../rulesets/segment2026_ruleset_setup.dcp)

  eCognition-ruleset brukt til prosjektoppsett, import av AR5 og aliasering av rasterlagene som segmenteringsrulesetet bruker.

- **Vedlegg 3:** [segment2026_ruleset_segment.dcp](../rulesets/segment2026_ruleset_segment.dcp)

  eCognition-ruleset med de separate segmenteringsløpene Segment A og Segment B, inkludert sletting av nivåer, AR5-segmentering, terskelverdier, multiresolution-parametere, lagvekter og eksportinnstillinger.

- **Kilder brukt i oppdateringen:**

  Vedlagt teknisk rapport fra 2025, segment2026_ruleset_setup.dcp, segment2026_ruleset_segment.dcp og cir_topo_stack_gpu.py.
