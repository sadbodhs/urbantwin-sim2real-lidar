# configs
- `pcdet/`: the model configs (15 LUMPI, 13 V2X-Real).
- `starter_kit_mirror/`: where the official kit's frame lists go (not redistributed).
- `lumpi_rig.json`: the five-sensor LUMPI Measurement 4 rig through which the walkers and riders are ray-cast into the UT-LUMPI training frames (`utl/synth_ped.py`, `lumpi_rig()`).
  - Per sensor: `device` and `session` (the LUMPI sensor and recording-session ids), `origin` (x, y, z in metres, in the LUMPI frame), `elev_deg` (beam elevation angles in degrees) and `n_az` (azimuth steps per turn).
  - Source: the beam tables come from LUMPI's public sensor metadata, and the five sensor positions were solved from the per-point range values of one real frame (Measurement 4, frame 0, far from any forbidden frame); no labels were used. This is item 5 of the LUMPI "Real-data use" section in `lumpi/README.md`.
  - The source data, LUMPI (Institute of Cartography and Geoinformatics, Leibniz University Hannover), is licensed CC BY-NC 3.0.
