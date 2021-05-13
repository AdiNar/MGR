# Short description
The following code was used to run simulations for my master thesis in computer science.

# Description
This simulator was used to make comparison between ApAlg variations and
benchmark consisting of various list schedulers.
It produces boxplot charts with makespans normalized by known
lower bounds for the optimal schedule length.
It also produces boxplots with algorithms running times.

# Run
First install dependencies (virtual environment is recommended, Python 3.8 is required):
```
pip install -r requirements.txt
```
Then run `main.py` script. 
```
./main.py
```
will run script with default parameters.

Possible adjustments are:
```
./main.py [-h] [--seed SEED] [--reps REPS] [--machines MACHINES [MACHINES ...]] [--jobs JOBS [JOBS ...]]
          [--output-dir OUTPUT_DIR] [--check-assertions] [--algorithms ALGORITHMS [ALGORITHMS ...]]

```
Check `./main.py --help` for details.
