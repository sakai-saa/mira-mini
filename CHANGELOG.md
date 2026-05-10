# Changelog

## MIRA mini public v0.1.3-buttonstyle

- Changed CSV download button text color to match the MIRA mini brown color palette.
- Changed CSV download button background to a white / cream tone.
- Added a pink gradient hover effect for CSV download buttons.
- Added a light hover effect for standard buttons.

## MIRA mini public v0.1.2

- Removed incomplete HTML div structures that caused empty Streamlit cards to appear.
- Fixed duplicated conversation log display in the research details section.
- Added `st.session_state.last_result` to preserve measurement results.
- Added `on_click="ignore"` to reduce UI state disruption during CSV download.
- Separated conversation CSV and measurement summary CSV.
- Added explanatory columns to the measurement summary CSV.
- Improved metric text visibility.