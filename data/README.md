# Data

The analysis workbook is not committed here.

The notebooks expect a project folder holding `data3.xlsx` along with the `Tables/` and
`final run/tables/` folders that the pipeline writes into. Point the code at it with:

```bash
set HEADWAY_BASE_DIR=D:\Headway            # Windows
export HEADWAY_BASE_DIR=/path/to/Headway   # macOS / Linux
```

Each row is one follower-leader observation from the videographic surveys: the target vehicle's
time headway together with its speed, speed difference, lateral placement, leader class, site
and flow covariates.
