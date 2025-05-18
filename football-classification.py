
import pandas as pd
import numpy as np

teams_clean_col = ['fixture_id', 'team_id', 'team_name', 'fouls','yellow_cards', 'red_cards','ball_possession','total_passes','completed_passes', 'pass_percentage', 'assists', 'goals','expected_goals', 'shots_on_goal', 'shots_off_goal', 'shots_insidebox','shots_outsidebox', 'total_shots', 'blocked_shots','corner_kicks', 'offsides']
stats_clean_col = ['fixture_id', 'fixture_date', 'fixture_referee', 'league_id', 'league_name', 'teams_home_id', 'teams_home_name', 'teams_away_id', 'teams_away_name']
players_clean_col = ['fixture_id', 'team_id', 'team_name', 'player_id', 'player_name','fouls_drawn','fouls_committed', 'yellow_cards', 'red_cards', 'game_minutes']

fix_teams_clean = pd.read_csv("fixture_stats_teams_clean.csv",usecols=teams_clean_col)
fix_clean = pd.read_csv("fixture_stats_clean.csv", usecols=stats_clean_col)
fix_players_clean = pd.read_csv("fixture_stats_players_clean.csv",usecols=players_clean_col)
season_standings = pd.read_csv("season_standings_clean.csv",usecols=['team_id','rank','form','played_all','wins_all', 'draws_all', 'losses_all', 'goals_for_all','last_updated'])

season_standings['wins'] = season_standings['form'].str.count('W').replace(np.nan, 0)
season_standings['draws'] = season_standings['form'].str.count('D').replace(np.nan, 0)
season_standings['losses'] = season_standings['form'].str.count('L').replace(np.nan, 0)
season_standings=season_standings.drop(columns=['form'])

season_standings['win_ratio'] = season_standings['wins_all'] / season_standings['played_all'].replace(np.nan, 0)
season_standings['draw_ratio'] = season_standings['draws_all'] / season_standings['played_all'].replace(np.nan, 0)
season_standings['loss_ratio'] = season_standings['losses_all'] / season_standings['played_all'].replace(np.nan, 0)
season_standings['goal_per_game'] = season_standings['goals_for_all'] / season_standings['played_all'].replace(np.nan, 0)
season_standings=season_standings.drop(columns=['wins_all','draws_all','losses_all','goals_for_all','played_all'])

pd.to_datetime(fix_clean['fixture_date'])
pd.to_datetime(season_standings['last_updated'])

# team ids are actual identifiers (not like player_ids)
fix_teams_clean.groupby(['team_id', 'team_name']).size().reset_index().team_name.value_counts()

# fixture_date added to the players statistics for further data preparation purposes: fix_players_clean
fix_players_clean = fix_players_clean.merge(fix_clean[['fixture_id', 'fixture_date']], on='fixture_id', how='left')
# fix_players_clean = fix_players_clean_og.copy()

# Összes kapott lap kiszámítása minden játékosra: fix_players_clean
run_mode = 'yellow' #(total, yellow, red)
if run_mode == 'total':
  fix_players_clean['total_cards'] = fix_players_clean.apply(lambda x: x['yellow_cards'] + x['red_cards'] * 2 if x['yellow_cards'] != 2 else 3, axis=1)
elif run_mode == 'yellow':
  fix_players_clean['total_cards'] = fix_players_clean['yellow_cards']
elif run_mode == 'red':
  fix_players_clean['total_cards'] = fix_players_clean['red_cards']
else:
  raise ValueError("Invalid run_mode. Must be 'total', 'yellow', or 'red'.")
fix_players_clean.drop(['yellow_cards', 'red_cards'], axis=1, inplace=True)


# Játékosok kidobása ha game_minutes isna: fix_players_clean
# Ezek a pályán kívül kapott lapok amik nem számítanak bele az összes lap számába a fogadóirodáknál
# fix_players_clean[(fix_players_clean['game_minutes'].isna()) & (fix_players_clean['total_cards'] != 0)].sort_values(by=['total_cards', 'fixture_date'], ascending=[False, False])

fix_players_clean.drop(fix_players_clean[fix_players_clean["game_minutes"].isna()].index, inplace=True)
fix_players_clean["game_minutes"].isna().sum()

# Összes kapott lap kiszámításra minden csapatra minden meccsen: fix_total_cards
fix_total_cards = fix_players_clean.groupby(['fixture_id', 'team_id'])['total_cards'].sum().reset_index()
fix_total_cards = fix_total_cards.merge(fix_clean[['fixture_id', 'fixture_date']], on='fixture_id', how='left')

# Sort by team and fixture_date to ensure past matches come first
fix_total_cards = fix_total_cards.sort_values(by=["team_id", "fixture_date"])

fix_total_cards['matches_last_14_days'] = 0
fix_total_cards['fixture_date'] = pd.to_datetime(fix_total_cards['fixture_date'])

for index, row in fix_total_cards.iterrows():
    team = row['team_id']
    date = row['fixture_date']

    # Filter past fixtures for this team in the last 14 days
    past_games = fix_total_cards[(fix_total_cards['team_id'] == team) &
                           (fix_total_cards['fixture_date'] < date) &
                           (fix_total_cards['fixture_date'] >= date - pd.Timedelta(days=14))]

    # Store count
    fix_total_cards.at[index, 'matches_last_14_days'] = len(past_games)


# Compute past average total cards for each team
fix_total_cards["avg_total_cards"] = (
    fix_total_cards.groupby("team_id")["total_cards"]
    .expanding()
    .mean()
    .shift()  # Shift to exclude current row from the average
    .reset_index(level=0, drop=True)
)

# Minden csapat legkorábbi meccsét dropoljuk, mert az avg_total_cards értéke ott nem megállapítható
fix_total_cards = fix_total_cards.drop(fix_total_cards.groupby('team_id')['fixture_date'].idxmin())

fix_total_cards.head()

# Meccsek adataihoz hozzá adjuk a csapatok átlagos kártyáit: fix_total_cards (és cards_tmp)
# Hazai hozzáadása
cards_tmp = fix_clean.merge(fix_total_cards.drop("fixture_date", axis=1), on="fixture_id", how="inner")
cards_tmp = cards_tmp[cards_tmp["team_id"] == cards_tmp["teams_home_id"]]
cards_tmp = cards_tmp.rename(columns={"total_cards": "home_total_cards_value", "avg_total_cards": "avg_home_total_cards",'matches_last_14_days': 'home_matches_last_14_days'})
cards_tmp = cards_tmp.drop(columns=["team_id"])

# Vendég kártyáinak hozzáadása
cards_tmp = cards_tmp.merge(fix_total_cards.drop("fixture_date", axis=1), on="fixture_id", how="inner")
cards_tmp = cards_tmp[cards_tmp["team_id"] == cards_tmp["teams_away_id"]]
cards_tmp = cards_tmp.rename(columns={"total_cards": "away_total_cards_value", "avg_total_cards": "avg_away_total_cards", 'matches_last_14_days': 'away_matches_last_14_days'})
cards_tmp = cards_tmp.drop(columns=["team_id"])

cards_tmp["total_cards_value"] = cards_tmp["home_total_cards_value"] + cards_tmp["away_total_cards_value"]
cards_tmp

fix_total_cards = cards_tmp[['fixture_id', 'fixture_referee', 'fixture_date', 'home_total_cards_value', 'avg_home_total_cards','home_matches_last_14_days', 'away_total_cards_value', 'avg_away_total_cards', 'away_matches_last_14_days', 'total_cards_value']]

# Sort by referee and fixture_date to ensure past matches come first
fix_total_cards = fix_total_cards.sort_values(by=["fixture_referee", "fixture_date"])

# Compute past average total cards for each referee
fix_total_cards["avg_referee_cards"] = (
    fix_total_cards.groupby("fixture_referee")["total_cards_value"]
    .expanding()
    .mean()
    .shift()  # Shift to exclude current row from the average
    .reset_index(level=0, drop=True)
)

# Minden referee legkorábbi meccsét dropoljuk, mert az avg_referee_cards értéke ott nem megállapítható
fix_total_cards = fix_total_cards.drop(fix_total_cards.groupby('fixture_referee')['fixture_date'].idxmin())

fix_total_cards.drop(fix_total_cards[fix_total_cards["fixture_referee"].isna()].index, inplace=True)

# Összes falt kiszámítása minden csapatra meccsenként: fix_total_fouls
fix_total_fouls = fix_teams_clean[["fixture_id", "team_id", "fouls","ball_possession",'total_passes','completed_passes', 'pass_percentage', 'assists', 'goals','expected_goals', 'shots_on_goal', 'shots_off_goal', 'shots_insidebox','shots_outsidebox', 'total_shots', 'blocked_shots','corner_kicks', 'offsides']]
fix_total_fouls.drop(fix_total_fouls[fix_total_fouls["fouls"] == 0].index, inplace=True)
fix_total_fouls = fix_total_fouls.merge(fix_clean[['fixture_id', 'fixture_date']], on='fixture_id', how='inner')

# Sort by team and fixture_date to ensure past matches come first
fix_total_fouls = fix_total_fouls.sort_values(by=["team_id", "fixture_date"])
fix_total_fouls['avg_possession_before'] = 0.0
fix_total_fouls["avg_possession_before"] = (
    fix_total_fouls.groupby("team_id")["ball_possession"]
    .expanding()
    .mean()
    .shift()  # Shift to exclude current row from the average
    .reset_index(level=0, drop=True)
)

# Compute past average total cards for each team
fix_total_fouls["avg_total_fouls"] = (
    fix_total_fouls.groupby("team_id")["fouls"]
    .expanding()
    .mean()
    .shift()  # Shift to exclude current row from the average
    .reset_index(level=0, drop=True)
)

avg_cols=['total_passes','completed_passes', 'pass_percentage', 'assists', 'goals','expected_goals', 'shots_on_goal', 'shots_off_goal', 'shots_insidebox','shots_outsidebox', 'total_shots', 'blocked_shots','corner_kicks', 'offsides']
for col in avg_cols:
  fix_total_fouls[f'avg_{col}_before']=0.0
  fix_total_fouls[f'avg_{col}_before'] = (
      fix_total_fouls.groupby("team_id")[col]
      .expanding()
      .mean()
      .shift()
      .reset_index(level=0, drop=True))
formatted_avg_cols = [f'avg_{col}_before' for col in avg_cols]

# Minden csapat legkorábbi meccsét dropoljk, mert az avg_total_fouls értéke ott nem megállapítható
fix_total_fouls = fix_total_fouls.drop(fix_total_fouls.groupby('team_id')['fixture_date'].idxmin())

# Fölösleges oszlopok elhagyása
fix_total_fouls = fix_total_fouls.drop(columns=['fixture_date',"fouls","ball_possession",'total_passes','completed_passes', 'pass_percentage', 'assists', 'goals','expected_goals', 'shots_on_goal', 'shots_off_goal', 'shots_insidebox','shots_outsidebox', 'total_shots', 'blocked_shots','corner_kicks', 'offsides'])

def calculate_avg_cards_per_90(df):
    # Sort by fixture_id to ensure chronological order
    df = df.sort_values(by=['player_name', 'fixture_date'])

    # Create new columns for storing cumulative stats
    df['cumulative_cards'] = 0
    df['cumulative_fouls'] = 0
    df['cumulative_minutes'] = 0
    df['avg_cards_per_90'] = 0.0
    df['avg_fouls_per_90'] = 0.0

    # Group by player_name and iterate to compute averages from past matches
    grouped = df.groupby('player_name')

    for player_name, player_df in grouped:
        cumulative_cards = 0
        cumulative_fouls = 0
        cumulative_minutes = 0

        for idx, row in player_df.iterrows():
            # Compute averages from past matches
            if cumulative_minutes > 90:
                avg_cards = (cumulative_cards / cumulative_minutes) * 90
                avg_fouls = (cumulative_fouls / cumulative_minutes) * 90
            else:
                avg_cards = 0
                avg_fouls = 0

            # Store calculated averages
            df.at[idx, 'avg_cards_per_90'] = avg_cards
            df.at[idx, 'avg_fouls_per_90'] = avg_fouls

            # Update cumulative stats
            cumulative_cards += row['total_cards']
            cumulative_fouls += row['fouls_committed']
            cumulative_minutes += row['game_minutes']

    return df

# Játékosok átlagos lap- és falt-számának meghatározás: fix_players_clean
fix_players_clean = calculate_avg_cards_per_90(fix_players_clean)

# keep only the 22 most played players for all the fixtures
fix_players_clean_22 = fix_players_clean.groupby('fixture_id').apply(lambda x: x.nlargest(22, 'game_minutes')).reset_index(drop=True)

# Négy topliga kiszűrése: fix_player_topleagues, fix_player_topleagues22
top_leagues = ["La Liga", "Serie A", "Bundesliga", "Premier League"]
fix_clean = fix_clean[fix_clean["league_name"].isin(top_leagues)]

teams_to_keep = list(set(list(fix_clean.teams_home_name.unique())+list(fix_clean.teams_away_name.unique())))

print(len(fix_players_clean))
fix_player_topleagues = fix_players_clean[fix_players_clean['team_name'].isin(teams_to_keep)]
fix_player_topleagues22 = fix_players_clean_22[fix_players_clean_22['team_name'].isin(teams_to_keep)]
print(len(fix_player_topleagues))

# Fix_player_topleagues de csak 22 legtöbbet játszó játékossal
fix_player_topleagues22['player_num'] = fix_player_topleagues22.groupby('fixture_id').cumcount() + 1

pivoted_fix_players_clean_final = fix_player_topleagues22.pivot_table(
    index='fixture_id',
    columns='player_num',
    values=['avg_cards_per_90', 'avg_fouls_per_90']
)

pivoted_fix_players_clean_final.columns = [
    f'player_{num}_avg_cards_per_90' if stat == 'avg_cards_per_90'
    else f'player_{num}_avg_fouls_per_90'
    for stat, num in pivoted_fix_players_clean_final.columns
]

pivoted_fix_players_clean_final.reset_index(inplace=True)
# pivoted_fix_players_clean_final

pivoted_fix_players_clean_final.fillna(0, inplace=True)

card_columns = [f'player_{num}_avg_cards_per_90' for num in range(1, 23)]
foul_columns = [f'player_{num}_avg_fouls_per_90' for num in range(1, 23)]

# Átlagos foul és lapszám kiszámolása minden játszó játékosra
fix_player_topleagues = fix_player_topleagues.groupby('fixture_id')[['avg_cards_per_90', 'avg_fouls_per_90']].sum().reset_index()

# Adding to the final data table the teams' card statistics
home_cols = {f'avg_{col}_before': f'avg_home_{col}_before' for col in avg_cols}
away_cols = {f'avg_{col}_before': f'avg_away_{col}_before' for col in avg_cols}

# for home team
merged = fix_clean.merge(fix_total_fouls, on="fixture_id", how="inner")
merged = merged[merged["team_id"] == merged["teams_home_id"]]
merged = merged.rename(columns={"avg_total_fouls": "avg_home_team_fouls","avg_possession_before":"avg_home_possession_before"} | home_cols)
merged = merged.drop(columns=["team_id"])

# for away team
merged = merged.merge(fix_total_fouls, on="fixture_id", how="inner")
merged = merged[merged["team_id"] == merged["teams_away_id"]]
merged = merged.rename(columns={"avg_total_fouls": "avg_away_team_fouls","avg_possession_before":"avg_away_possession_before"} | away_cols)
merged = merged.drop(columns=["team_id"])

# Adding to the final data table the fix_total_cards
merged = merged.merge(fix_total_cards.drop(['fixture_date', 'fixture_referee'], axis=1), on="fixture_id", how="inner")

# Adding to the final table the sum of player statistics and the separate player statistics
merged = merged.merge(fix_player_topleagues, on="fixture_id", how="inner")
merged = merged.merge(pivoted_fix_players_clean_final, on="fixture_id", how="inner")
merged

# Needed in the final table:
#fixture referee, league_name, teams_home_name, teams_away_name, avg_home_team_fouls, avg_home_team_cards, avg_away_team_fouls, avg_away_teasm_cards, [avg_total_cards_player_{1-22}, avg_total_fouls_player_{1_22},] total_cards

final_cols = ["fixture_referee","fixture_date", "league_name", "teams_home_name", "teams_away_name", "avg_home_team_fouls", "avg_away_team_fouls","avg_home_total_cards","avg_away_total_cards", "avg_referee_cards", "avg_cards_per_90", "avg_fouls_per_90", 'home_matches_last_14_days','away_matches_last_14_days',"avg_home_possession_before","avg_away_possession_before", "total_cards_value"]+card_columns+foul_columns

merged_final = merged

# Check if there is any nan value in the final table
nanseries = merged.isnull().sum() / merged.shape[0] * 100
nullnanseries = nanseries[nanseries > 0]
print(nanseries[nanseries > 0])
len(nullnanseries)

merged_final.info()

# saving merged_final based on run mode
if run_mode == 'total':
  total_final = merged_final.copy()
elif run_mode == 'yellow':
  yellow_final = merged_final.copy()
elif run_mode == 'red':
  red_final = merged_final.copy()
else:
  raise ValueError("Invalid run_mode. Must be 'total', 'yellow', or 'red'.")

run_mode = 'yellow' #(total, yellow, red)
if run_mode == 'total':
  model_input_df = total_final.copy()
elif run_mode == 'yellow':
  model_input_df = yellow_final.copy()
elif run_mode == 'red':
  model_input_df = red_final.copy()
else:
  raise ValueError("Invalid run_mode. Must be 'total', 'yellow', or 'red'.")

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 500)

df = model_input_df.sort_values(by="fixture_date",ascending=True)
df.drop_duplicates(['fixture_date', 'teams_home_name', 'teams_away_name'])[['fixture_date', 'teams_home_name', 'teams_away_name', 'total_cards_value']].head(20)

df = model_input_df.sort_values(by="fixture_date",ascending=True)
df.drop_duplicates(['fixture_date', 'teams_home_name', 'teams_away_name'])[['fixture_date', 'teams_home_name', 'teams_away_name', 'total_cards_value']].head(20)

# Get the unique teams in both 'team_home_name' and 'team_away_name'
home_teams = set(df['teams_home_name'].unique())
away_teams = set(df['teams_away_name'].unique())

# Find teams that are only in one of the columns (either home or away)
teams_only_home = home_teams - away_teams
teams_only_away = away_teams - home_teams

print("Teams only in 'teams_home_name':", teams_only_home)
print("Teams only in 'teams_away_name':", teams_only_away)

df_sorted = df.sort_values(by='fixture_date')
df_last = df_sorted.drop_duplicates(subset='teams_home_name', keep='last')

cols = ['avg_home_possession_before', 'avg_home_team_fouls',
       'avg_home_total_passes_before', 'avg_home_completed_passes_before',
       'avg_home_pass_percentage_before', 'avg_home_assists_before',
       'avg_home_goals_before', 'avg_home_expected_goals_before',
       'avg_home_shots_on_goal_before', 'avg_home_shots_off_goal_before',
       'avg_home_shots_insidebox_before', 'avg_home_shots_outsidebox_before',
       'avg_home_total_shots_before', 'avg_home_blocked_shots_before',
       'avg_home_corner_kicks_before', 'avg_home_total_cards']
X = df_last[cols]

import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

# Skálázás
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

umap_neighbors = [5, 10, 15, 30]
kmeans_clusters = [2, 3, 4, 5, 6]

best_score = -1
best_config = {}

for n in umap_neighbors:
    umap_model = umap.UMAP(n_neighbors=n, min_dist=0.1, n_components=2, random_state=42)
    X_umap = umap_model.fit_transform(X_scaled)

    for k in kmeans_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_umap)

        sil = silhouette_score(X_umap, labels)
        db = davies_bouldin_score(X_umap, labels)
        ch = calinski_harabasz_score(X_umap, labels)

        print(f"UMAP n_neighbors={n}, KMeans n_clusters={k}")
        print(f"  Silhouette: {sil:.3f}, DB Index: {db:.3f}, CH Index: {ch:.2f}")

        if sil > best_score:
            best_score = sil
            best_config = {
                "n_neighbors": n,
                "n_clusters": k,
                "labels": labels,
                "X_umap": X_umap,
                "scores": (sil, db, ch)
            }

umap_model = umap.UMAP(n_neighbors=best_config["n_neighbors"],n_components=2, random_state=42)
X_scaled = umap_model.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=best_config["n_clusters"], random_state=42)
df_kmeans2 = df_last.copy()
df_kmeans2['cluster'] = kmeans.fit_predict(X_scaled)
res2 = df_kmeans2[['fixture_date', 'league_name','teams_home_name', 'teams_away_name']+cols+['total_cards_value']+['cluster']]

print(silhouette_score(X_scaled, df_kmeans2['cluster']))
print(davies_bouldin_score(X_scaled, df_kmeans2['cluster']))
print(calinski_harabasz_score(X_scaled, df_kmeans2['cluster']))

clustered_teams = (
    df_kmeans2
    .sort_values(by='home_total_cards_value', ascending=False)
    .groupby('cluster')['teams_home_name']
    .apply(list)
)
for cluster, teams in clustered_teams.items():
    top_5 = teams[:5]  # take only the first 5
    print(f"Cluster {cluster}: {top_5}")

# Compute average cards per cluster
cluster_aggressivity = {}

for cluster_id, team_list in clustered_teams.items():
    # Filter the dataframe to the teams in the current cluster
    cluster_df = df_kmeans2[df_kmeans2['teams_home_name'].isin(team_list)]

    # Compute mean total cards
    if not cluster_df.empty:
        avg_cards = cluster_df['home_total_cards_value'].mean()
    else:
        avg_cards = 0  # or np.nan if you want to skip

    cluster_aggressivity[cluster_id] = avg_cards

# Rank clusters by aggressivity (descending)
ranked = sorted(cluster_aggressivity.items(), key=lambda x: x[1], reverse=True)

# Display

print("\nCluster Ranking by Aggressivity (Total Cards):")
for rank, (cluster_id, avg_cards) in enumerate(ranked, 1):
    print(f"{rank}. Cluster {cluster_id} → Avg Cards: {avg_cards:.2f}")




