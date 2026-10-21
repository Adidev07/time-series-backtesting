# Model / system card

## Intended use

Forecast one step ahead using lag features and expanding-window backtesting. Educational local experiment; no production deployment is implied.

## Data

Synthetic arrays, manually written documents or replayed JSON. No real user records. Seeds and sample inputs are recorded in app.py and sample_input.json.

## Method

Construct fourteen historical lags and weekday sine/cosine features. Ridge regression uses centered training design matrices and solves a regularized linear system. Evaluate each fold using only earlier observations.

## Evaluation protocol

Select regularization on earlier chronological folds; reserve the last 80 days as a final rolling one-step test. Compare seasonal-naive predictions from seven days earlier.

## Known limits

Every test prediction may use previously observed test values, which is valid for rolling one-step forecasts, not an 80-step open-loop forecast. The series is synthetic.

## Verification status

Local tests and default experiment are executed during packaging. Live model generation is not exercised in the build. A successful validation result is limited to the checks implemented in code.

## Further work

Add real demand data, horizon-specific evaluation, holidays, prediction intervals and an explicit multi-step forecasting mode.
