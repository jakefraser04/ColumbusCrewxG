"""
update_season.py
Columbus Crew xG Dashboard — Season Updater
---------------------------------------------
Run this after every Columbus Crew match.

Usage:
    python update_season.py --season 2026
    python update_season.py --season 2026 --push
"""

# ── Imports ───────────────────────────────────────────────
# Each 'import' line loads an external library into memory.
# Think of it like opening a toolbox before you start working.

import os           # lets us create folders and work with file paths
import sys          # lets us exit the script early if something goes wrong
import json         # lets us read and write JSON files (our dashboard data format)
import argparse     # lets us accept command-line arguments like --push and --season
import subprocess   # lets us run git commands from inside Python
import warnings     # lets us suppress non-critical warning messages

import pandas as pd         # the core data manipulation library — DataFrames
import numpy as np          # numerical operations (math on arrays)
import matplotlib.pyplot as plt          # creates charts
import matplotlib.patches as mpatches   # creates legend color patches
import joblib                            # saves and loads our trained ML model

from itscalledsoccer.client import AmericanSoccerAnalysis
# ^ imports the ASA API client so we can fetch MLS match data


# ── Suppress non-critical warnings ───────────────────────
warnings.filterwarnings('ignore')
# ^ tells Python to hide minor warnings that don't affect our results


# ── Argument parser ───────────────────────────────────────
# This lets you pass options when running the script in the terminal.
# e.g. `python update_season.py --season 2026 --push`
parser = argparse.ArgumentParser()

# --season lets you specify which year to fetch (default is 2026)
parser.add_argument('--season', default='2026', help='Season year to fetch')

# --push is a flag — if you include it, the script pushes to GitHub
parser.add_argument('--push', action='store_true', help='Push to GitHub after updating')

args = parser.parse_args()   # actually reads the arguments you passed in
SEASON = args.season         # stores the season string e.g. '2026'


# ── Constants ─────────────────────────────────────────────
# These are values that don't change during the script.
# Using ALL_CAPS is a Python convention for constants.

CREW_ID   = 'mvzqoLZQap'        # Columbus Crew's unique ID in the ASA database
CREW_NAME = 'Columbus Crew'

CREW_GOLD  = '#F5C518'   # Crew gold hex color for charts
CREW_BLACK = '#14130F'   # Crew black hex color for chart backgrounds
CREW_GREY  = '#2A2820'   # Slightly lighter black for legend backgrounds

# Dictionary mapping result codes to their chart colors
# Dictionaries use {key: value} syntax — like a lookup table
RESULT_COLORS = {
    'W': '#27AE60',   # green for wins
    'D': '#E67E22',   # orange for draws
    'L': '#C0392B'    # red for losses
}


# ── File paths ────────────────────────────────────────────
# f-strings (f"...") let you embed variables inside strings
# {SEASON} gets replaced with '2026' (or whatever season you passed in)

ASSETS_DIR = f'assets/{SEASON}'          # e.g. 'assets/2026'
DATA_DIR   = f'dashboard/data/{SEASON}'  # e.g. 'dashboard/data/2026'
PROCESSED  = f'data/processed/{SEASON}'  # e.g. 'data/processed/2026'
MODEL_PATH = 'models/xg_model.pkl'       # our trained model (same for all seasons)
META_PATH  = 'models/model_meta.json'    # model metadata (AUC, log loss, etc.)

# os.makedirs creates a folder if it doesn't already exist
# exist_ok=True means "don't crash if the folder is already there"
for d in [ASSETS_DIR, DATA_DIR, PROCESSED]:
    os.makedirs(d, exist_ok=True)


# ── Helper function ───────────────────────────────────────
# Functions are reusable blocks of code. 'def' defines a new function.
# log() just prints a message with consistent formatting.
def log(msg):
    print(f'  {msg}')   # the f-string adds 2 spaces before every message


# ══════════════════════════════════════════════════════════
# STEP 1: Load the trained xG model
# ══════════════════════════════════════════════════════════
def load_model():
    log(f'Loading xG model...')

    # joblib.load reads the .pkl file and reconstructs the scikit-learn model
    # A .pkl (pickle) file is Python's format for saving objects to disk
    model = joblib.load(MODEL_PATH)

    # open() opens a file — 'with' ensures it closes automatically when done
    # json.load() parses the JSON text into a Python dictionary
    with open(META_PATH) as f:
        meta = json.load(f)

    log(f'Model ready — AUC: {meta["auc"]}  trained on {meta["training_shots"]:,} shots')
    # :, in the format string adds comma separators to large numbers (e.g. 10,000)

    return model   # 'return' sends the model back to whoever called this function


# ══════════════════════════════════════════════════════════
# STEP 2: Fetch Columbus Crew match data & shots from ASA
# ══════════════════════════════════════════════════════════
def fetch_crew_data():
    log(f'Fetching {CREW_NAME} {SEASON} data from American Soccer Analysis...')
    api = AmericanSoccerAnalysis()

    # Call the API to get match xG data
    raw = api.get_game_xgoals(
        leagues='mls',
        season_name=SEASON,
        team_ids=CREW_ID
    )

    teams = api.get_teams(leagues='mls')
    team_lookup = teams.set_index('team_id')['team_name'].to_dict()

    matches = []
    for _, row in raw.iterrows():
        is_home = row['home_team_id'] == CREW_ID

        xg_for    = float(row['home_team_xgoals'] if is_home else row['away_team_xgoals'])
        xg_ag     = float(row['away_team_xgoals'] if is_home else row['home_team_xgoals'])
        goals_for = int(row['home_goals'] if is_home else row['away_goals'])
        goals_ag  = int(row['away_goals'] if is_home else row['home_goals'])
        opp_id    = row['away_team_id'] if is_home else row['home_team_id']

        if goals_for > goals_ag:    result = 'W'
        elif goals_for == goals_ag: result = 'D'
        else:                       result = 'L'

        if xg_for > xg_ag + 0.3:   xg_result = 'xW'
        elif xg_ag > xg_for + 0.3: xg_result = 'xL'
        else:                       xg_result = 'xD'

        deserved = (
            (result == 'W' and xg_result == 'xW') or
            (result == 'D' and xg_result == 'xD') or
            (result == 'L' and xg_result == 'xL')
        )

        matches.append({
            'date':       str(row['date_time_utc'])[:10],
            'opponent':   team_lookup.get(opp_id, opp_id),
            'venue':      'Home' if is_home else 'Away',
            'goals_for':  goals_for,
            'goals_ag':   goals_ag,
            'xg_for':     round(xg_for, 2),
            'xg_ag':      round(xg_ag, 2),
            'xg_diff':    round(xg_for - xg_ag, 2),
            'result':     result,
            'xg_result':  xg_result,
            'deserved':   deserved,
        })

    crew = pd.DataFrame(matches)
    crew = crew.sort_values('date').reset_index(drop=True)

    crew['cumulative_xg']    = crew['xg_for'].cumsum().round(2)
    crew['cumulative_xga']   = crew['xg_ag'].cumsum().round(2)
    crew['cumulative_goals'] = crew['goals_for'].cumsum()

    crew['rolling_xg_for'] = crew['xg_for'].rolling(5, min_periods=1).mean()
    crew['rolling_xg_ag']  = crew['xg_ag'].rolling(5, min_periods=1).mean()

    log(f'Fetched {len(crew)} matches')
    return crew


# ══════════════════════════════════════════════════════════
# STEP 3: Build and save all charts
# ══════════════════════════════════════════════════════════
def build_charts(crew):
    log('Building charts...')

    # range(len(crew)) creates [0, 1, 2, ... n-1]
    # This gives us the x-axis positions for our charts
    x = range(len(crew))

    # ── Chart 1: Cumulative xG vs Goals ───────────────────
    # plt.subplots() creates a figure and an axes object
    # figsize=(15, 5) sets width=15 inches, height=5 inches
    fig, ax = plt.subplots(figsize=(15, 5))
    fig.patch.set_facecolor(CREW_BLACK)   # outer figure background color
    ax.set_facecolor(CREW_BLACK)          # inner plot area background color

    # fill_between() shades the area under a line
    # alpha= controls transparency (0=invisible, 1=solid)
    ax.fill_between(x, crew['cumulative_xg'], alpha=0.12, color=CREW_GOLD)

    # ax.plot() draws a line — linewidth controls thickness, label shows in legend
    ax.plot(x, crew['cumulative_xg'],
            color=CREW_GOLD, linewidth=2.5, label='Cumulative xG')
    ax.plot(x, crew['cumulative_goals'],
            color='white', linewidth=2, linestyle='--', label='Actual Goals')

    # Loop through each match to add a colored dot per result
    for i, row in crew.iterrows():
        ax.scatter(i, row['cumulative_xg'],
                   color=RESULT_COLORS[row['result']], s=50, zorder=5)
        # s= controls dot size, zorder=5 draws it on top of other elements

    # mpatches.Patch() creates a colored rectangle for the legend
    patches = [mpatches.Patch(color=v, label=k) for k, v in RESULT_COLORS.items()]
    # ^ list comprehension: a compact way to build a list using a loop

    leg1 = ax.legend(handles=patches, loc='upper left',
                     facecolor=CREW_GREY, labelcolor='white',
                     title='Result', title_fontsize=9)
    ax.add_artist(leg1)   # add_artist keeps the first legend visible when adding a second
    ax.legend(loc='upper left', bbox_to_anchor=(0, 0.78),
              facecolor=CREW_GREY, labelcolor='white')

    # set_xticks() positions the tick marks on the x-axis
    ax.set_xticks(range(len(crew)))
    # set_xticklabels() sets the text labels at each tick
    # \n inserts a line break in the label
    ax.set_xticklabels(
        [f"{r['opponent'][:6]}\n{r['venue'][0]}" for _, r in crew.iterrows()],
        fontsize=7, color='#888'
    )
    ax.tick_params(axis='y', colors='#888')  # y-axis tick color
    ax.spines[:].set_visible(False)           # hides the border around the plot
    ax.set_title(f'Columbus Crew {SEASON} — Cumulative xG vs Actual Goals',
                 color='white', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Goals / xG', color='#888')

    plt.tight_layout()   # automatically adjusts spacing so nothing overlaps
    # savefig() saves the chart as a PNG image file
    plt.savefig(f'{ASSETS_DIR}/xg_trend.png', dpi=150,
                bbox_inches='tight', facecolor=CREW_BLACK)
    plt.close()   # close() frees memory — important when making multiple charts
    log('  Saved xg_trend.png')

    # ── Chart 2: xG Differential ──────────────────────────
    fig, ax = plt.subplots(figsize=(15, 5))
    fig.patch.set_facecolor(CREW_BLACK)
    ax.set_facecolor(CREW_BLACK)

    # list comprehension to get a color for each bar based on match result
    bar_colors = [RESULT_COLORS[r] for r in crew['result']]
    ax.bar(x, crew['xg_diff'], color=bar_colors, alpha=0.85, width=0.7)

    # axhline() draws a horizontal reference line across the chart
    ax.axhline(0, color='#555', linewidth=0.8)

    # Add asterisk to matches where the result didn't match xG
    for i, row in crew.iterrows():
        if not row['deserved']:   # 'not' inverts a boolean (True becomes False)
            ax.annotate('*', (i, row['xg_diff']),
                        ha='center',
                        va='bottom' if row['xg_diff'] >= 0 else 'top',
                        color='white', fontsize=10)

    ax.set_xticks(range(len(crew)))
    ax.set_xticklabels([r['opponent'][:6] for _, r in crew.iterrows()],
                       fontsize=7, color='#888')
    ax.tick_params(axis='y', colors='#888')
    ax.spines[:].set_visible(False)
    ax.set_title(f'Columbus Crew {SEASON} — xG Differential Per Match\n(* = result did not match xG)',
                 color='white', fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel('xG For − xG Against', color='#888')
    patches = [mpatches.Patch(color=v, label=k) for k, v in RESULT_COLORS.items()]
    ax.legend(handles=patches, facecolor=CREW_GREY, labelcolor='white')
    plt.tight_layout()
    plt.savefig(f'{ASSETS_DIR}/xg_differential.png', dpi=150,
                bbox_inches='tight', facecolor=CREW_BLACK)
    plt.close()
    log('  Saved xg_differential.png')

    # ── Chart 3: Rolling form ──────────────────────────────
    fig, ax = plt.subplots(figsize=(15, 5))
    fig.patch.set_facecolor(CREW_BLACK)
    ax.set_facecolor(CREW_BLACK)

    # fill_between with 'where' only shades where the condition is True
    ax.fill_between(x, crew['rolling_xg_for'], crew['rolling_xg_ag'],
                    where=crew['rolling_xg_for'] >= crew['rolling_xg_ag'],
                    alpha=0.2, color=CREW_GOLD)
    ax.fill_between(x, crew['rolling_xg_for'], crew['rolling_xg_ag'],
                    where=crew['rolling_xg_for'] < crew['rolling_xg_ag'],
                    alpha=0.2, color='#C0392B')
    ax.plot(x, crew['rolling_xg_for'], color=CREW_GOLD, linewidth=2.5, label='5-match avg xG For')
    ax.plot(x, crew['rolling_xg_ag'],  color='#C0392B',  linewidth=2.5, label='5-match avg xG Against')
    ax.set_xticks(range(len(crew)))
    ax.set_xticklabels([r['opponent'][:6] for _, r in crew.iterrows()], fontsize=7, color='#888')
    ax.tick_params(axis='y', colors='#888')
    ax.spines[:].set_visible(False)
    ax.set_title(f'Columbus Crew {SEASON} — Rolling 5-Match xG Form',
                 color='white', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Average xG (5-match rolling)', color='#888')
    ax.legend(facecolor=CREW_GREY, labelcolor='white')
    plt.tight_layout()
    plt.savefig(f'{ASSETS_DIR}/rolling_xg_form.png', dpi=150,
                bbox_inches='tight', facecolor=CREW_BLACK)
    plt.close()
    log('  Saved rolling_xg_form.png')

    # ── Chart 4: Home vs Away scatter ─────────────────────
    # Boolean indexing: filters rows where the 'venue' column equals 'Home'
    home = crew[crew['venue'] == 'Home']
    away = crew[crew['venue'] == 'Away']

    # plt.subplots(1, 2) creates 1 row, 2 columns of charts side by side
    # axes is a list of two ax objects
    fig, axes = plt.subplots(1, 2, figsize=(9, 7))
    fig.patch.set_facecolor(CREW_BLACK)

    # zip() pairs two lists together so we can loop through both at once
    for ax, subset, label in zip(axes, [home, away], ['Home', 'Away']):
        ax.set_facecolor(CREW_BLACK)
        if len(subset) == 0:   # handle case where no home/away matches yet
            ax.set_title(f'{label} — No data yet', color='white')
            continue           # 'continue' skips the rest of this loop iteration

        ax.scatter(subset['xg_for'], subset['xg_ag'],
                   c=[RESULT_COLORS[r] for r in subset['result']],
                   s=80, zorder=5, edgecolors='white', linewidth=0.4)

        # Draw diagonal line — points above = opponent had better chances
        lim = max(subset['xg_for'].max(), subset['xg_ag'].max()) + 0.3
        ax.plot([0, lim], [0, lim], '--', color='#555', linewidth=1)

        for _, row in subset.iterrows():
            ax.annotate(row['opponent'][:5], (row['xg_for'], row['xg_ag']),
                        textcoords='offset points', xytext=(5, 3),
                        fontsize=6, color='#aaa')

        ax.set_xlabel('Crew xG For', color='#888')
        ax.set_ylabel('Crew xG Against', color='#888')
        ax.tick_params(colors='#888')
        ax.spines[:].set_visible(False)
        ax.set_title(f'{label} Matches', color='white', fontweight='bold')

    patches = [mpatches.Patch(color=v, label=k) for k, v in RESULT_COLORS.items()]
    axes[1].legend(handles=patches, facecolor=CREW_GREY, labelcolor='white')
    plt.suptitle(f'Columbus Crew {SEASON} — xG For vs Against',
                 color='white', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{ASSETS_DIR}/home_away_xg.png', dpi=150,
                bbox_inches='tight', facecolor=CREW_BLACK)
    plt.close()
    log('  Saved home_away_xg.png')
    
    # ── Chart 5: Pitch Shot Map & xG ──────────────────────
    from mplsoccer import Pitch

    shots_path = f'data/processed/{SEASON}/crew_shots.csv'
    if os.path.exists(shots_path):
        df_shots = pd.read_csv(shots_path)

        if not df_shots.empty and 'x' in df_shots.columns and 'y' in df_shots.columns:
            # ASA pitch dimensions are typically standardized 0-100 or yard scales
            # Standard StatsBomb scale is 120 x 80. If ASA coordinates are 0-100, scale them:
            # df_shots['x_scaled'] = df_shots['x'] * 1.2
            # df_shots['y_scaled'] = df_shots['y'] * 0.8
            
            pitch = Pitch(
                pitch_type='custom', 
                pitch_length=100, 
                pitch_width=100,
                pitch_color=CREW_BLACK, 
                line_color='#444444', 
                half=True
            )
            fig, ax = pitch.draw(figsize=(9, 7))
            fig.patch.set_facecolor(CREW_BLACK)

            # Check outcome/goal identifier from ASA ('Goal' or is_goal == 1)
            is_goal = df_shots['result'].str.lower() == 'goal' if 'result' in df_shots.columns else df_shots['is_goal'] == 1
            goals = df_shots[is_goal]
            non_goals = df_shots[~is_goal]

            # Scatter Non-Goals (Yellow circles)
            pitch.scatter(
                non_goals['x'], non_goals['y'],
                s=non_goals['xg'] * 500 + 40,
                c=CREW_GOLD, alpha=0.65, edgecolors='white', linewidth=0.5,
                marker='o', label='Shot (No Goal)', ax=ax
            )

            # Scatter Goals (Green stars)
            pitch.scatter(
                goals['x'], goals['y'],
                s=goals['xg'] * 500 + 90,
                c='#2ecc71', alpha=0.95, edgecolors='white', linewidth=1.2,
                marker='*', label='Goal', ax=ax
            )

            ax.legend(loc='lower left', facecolor=CREW_GREY, edgecolor='none', labelcolor='white', fontsize=8)
            ax.set_title(f'Columbus Crew {SEASON} — Shot Map & xG Quality',
                         color='white', fontsize=12, fontweight='bold', pad=10)

            plt.tight_layout()
            plt.savefig(f'{ASSETS_DIR}/xG_heatmap.png', dpi=180,
                        bbox_inches='tight', facecolor=CREW_BLACK)
            plt.close()
            log('  Saved updated xG_heatmap.png with match shots')


# ══════════════════════════════════════════════════════════
# STEP 4: Export JSON files for the dashboard
# ══════════════════════════════════════════════════════════
def export_json(crew):
    log('Exporting JSON files...')

    # Count wins/draws/losses using boolean comparison
    # (crew['result'] == 'W') creates a True/False column
    # .sum() counts all the Trues (True = 1, False = 0)
    W = (crew['result'] == 'W').sum()
    D = (crew['result'] == 'D').sum()
    L = (crew['result'] == 'L').sum()

    # Filter for undeserved results using multiple conditions
    # & means AND (both conditions must be True)
    lucky   = crew[(crew['result'] == 'W') & (crew['xg_result'] == 'xL')]
    unlucky = crew[(crew['result'] == 'L') & (crew['xg_result'] == 'xW')]
    deserved = crew[crew['deserved']]

    # Convert entire DataFrame to a list of dictionaries for JSON export
    match_records = crew.to_dict(orient='records')

    # Write match data to JSON file
    # 'w' mode opens the file for writing (creates it if it doesn't exist)
    # json.dump() converts Python objects to JSON text
    # indent=2 makes the JSON human-readable with 2-space indentation
    # default=str converts any non-serializable values (like numpy types) to strings
    with open(f'{DATA_DIR}/match_xg.json', 'w') as f:
        json.dump(match_records, f, indent=2, default=str)

    # Build season summary dictionary
    summary = {
        'team':           CREW_NAME,
        'season':         SEASON,
        'matches':        len(crew),        # len() counts rows in the DataFrame
        'wins':           int(W),           # int() converts numpy int to Python int
        'draws':          int(D),
        'losses':         int(L),
        'total_xg_for':   round(float(crew['xg_for'].sum()), 2),
        'total_xg_ag':    round(float(crew['xg_ag'].sum()), 2),
        'total_goals':    int(crew['goals_for'].sum()),
        'xg_diff':        round(float(crew['xg_diff'].sum()), 2),
        'lucky_wins':     int(len(lucky)),
        'unlucky_losses': int(len(unlucky)),
        # Ternary operator prevents division by zero if no matches yet
        'deserved_pct':   round(len(deserved) / len(crew) * 100, 1) if len(crew) > 0 else 0
    }

    with open(f'{DATA_DIR}/season_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Also save as CSV for use in notebooks
    crew.to_csv(f'{PROCESSED}/crew_match_xg.csv', index=False)
    # index=False prevents pandas from writing the row numbers as a column

    log(f'Exported {len(match_records)} matches')
    log(f'Record: {W}W {D}D {L}L')
    log(f'xG For: {summary["total_xg_for"]}  |  xG Against: {summary["total_xg_ag"]}')


# ══════════════════════════════════════════════════════════
# STEP 5: Push updated files to GitHub
# ══════════════════════════════════════════════════════════
def push_to_github(crew):
    # Build a descriptive commit message from the most recent match
    last_match = crew.iloc[-1]
    # .iloc[-1] gets the last row by position (-1 = last item, like Python lists)

    result_word = {'W': 'Win', 'D': 'Draw', 'L': 'Loss'}[last_match['result']]
    commit_msg = (
        f"Update {SEASON} data — "
        f"Matchday {len(crew)}: "
        f"{result_word} vs {last_match['opponent']} "
        f"({last_match['goals_for']}-{last_match['goals_ag']}) "
        f"xG {last_match['xg_for']:.2f} vs {last_match['xg_ag']:.2f}"
    )

    log(f'Committing: "{commit_msg}"')

    # List of git commands to run in sequence
    commands = [
        ['git', 'add', f'assets/{SEASON}/', f'dashboard/data/{SEASON}/'],
        ['git', 'commit', '-m', commit_msg],
        ['git', 'push', 'origin', 'main']
    ]

    # subprocess.run() executes a terminal command from Python
    # capture_output=True captures stdout/stderr so we can check for errors
    # text=True returns output as a string instead of bytes
    for cmd in commands:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # returncode != 0 means the command failed
            print(f'  Git error: {result.stderr.strip()}')
            return   # stop the function early if git fails
    log('Pushed successfully!')
    log(f'Live at: https://jakefraser04.github.io/Crew2025xG/')


# ══════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════
# This block only runs when you execute the script directly.
# It won't run if another script imports this file as a module.
if __name__ == '__main__':

    print()
    print('=' * 54)
    print(f'  Columbus Crew xG Dashboard — {SEASON} Season Update')
    print('=' * 54)

    load_model()              # Step 1: verify model is ready
    crew = fetch_crew_data()  # Step 2: pull latest match data

    # Early exit if no matches have been played yet
    if len(crew) == 0:
        print(f'\n  No {SEASON} matches found yet. Try again after the first game.')
        sys.exit(0)   # sys.exit(0) ends the script — 0 means "exited cleanly"

    build_charts(crew)   # Step 3: regenerate all charts
    export_json(crew)    # Step 4: update JSON files for dashboard

    if args.push:
        push_to_github(crew)   # Step 5 (optional): push to GitHub
    else:
        log('Done! Add --push to push to GitHub automatically.')

    print('=' * 54)
    print()
