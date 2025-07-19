import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

_DATA = None

def get_url(
    league
):
    
    urls = {
        'ligue 1': 'https://fbref.com/en/comps/13/schedule/Ligue-1-Scores-and-Fixtures',
        'bundesliga': 'https://fbref.com/en/comps/20/schedule/Bundesliga-Scores-and-Fixtures',
        'serie a': 'https://fbref.com/en/comps/11/schedule/Serie-A-Scores-and-Fixtures',
        'laliga': 'https://fbref.com/en/comps/12/schedule/La-Liga-Scores-and-Fixtures',
        'premier league': 'https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures',
        'championship': 'https://fbref.com/en/comps/10/schedule/Championship-Scores-and-Fixtures',
        'eredivisie': 'https://fbref.com/en/comps/23/schedule/Eredivisie-Scores-and-Fixtures',
        'liga portugal': 'https://fbref.com/en/comps/32/schedule/Liga-Portuguesa-Scores-and-Fixtures',
        'brazil': 'https://fbref.com/en/comps/24/schedule/Serie-A-Scores-and-Fixtures',
        'spl': 'https://fbref.com/en/comps/40/Scottish-Premiership-Stats'
    }
    
    return urls.get(league)

def get_league_data(
    league
):
    """
    Fetch & parse FBref fixtures once, caching the result.
    Returns DataFrame with parsed scores & dates.
    """
    
    global _DATA
    
    if _DATA is not None:
    
        return _DATA

    url = get_url(league)
    
    if url is None:
    
        raise ValueError(f"Unknown league {league!r}")

    resp = requests.get(url)

    soup = BeautifulSoup(resp.text, 'html.parser')
    
    table = soup.find('table')
    
    rows = table.find_all('tr')
    
    data = []
    
    for tr in rows:
    
        cols = tr.find_all('td')
        
        if not cols:
        
            continue
        
        data.append([td.text.strip() for td in cols])

    cols = [
        'Wk',
        'Date',
        'Time',
        'Home',
        'xG_home',
        'Score',
        'xG_away',
        'Away',
        'Attendance',
        'Venue',
        'Referee',
        'Match Report',
        'Notes'
    ]
    
    df = pd.DataFrame(data, columns=cols)

    df['Score'] = df['Score'].str.replace('–', '-', regex=False).str.strip()
    
    parts = df['Score'].str.split('-', expand=True)
    
    df['home_score'] = pd.to_numeric(parts[0].str.strip(), errors='coerce')
    
    df['away_score'] = pd.to_numeric(parts[1].str.strip(), errors='coerce')
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    _DATA = df
    
    return df


def match_result(
    home_score, 
    away_score
):
    
    if home_score > away_score:
    
        return "Win"
  
    elif home_score < away_score:
    
        return "Loss"
    
    else:
    
        return "Draw"

def team_results(
    data
):
    """
    Vectorized home/away W-D-L counts.
    """
  
    df = data.dropna(subset=['home_score','away_score']).copy()

    
    home_res = np.select(
        [df.home_score > df.away_score,
         df.home_score == df.away_score],
        ['Win','Draw'],
        default='Loss'
    )
    
    away_res = np.select(
        [df.away_score > df.home_score,
         df.away_score == df.home_score],
        ['Win','Draw'],
        default='Loss'
    )
    
    df['home_res'], df['away_res'] = home_res, away_res

    home_counts = (
        df.groupby('Home')['home_res']
          .value_counts()
          .unstack(fill_value=0)
          .rename(columns={
              'Win':'Home_Wins',
              'Draw':'Home_Draws',
              'Loss':'Home_Losses'
          })
    )
   
    away_counts = (
        df.groupby('Away')['away_res']
          .value_counts()
          .unstack(fill_value=0)
          .rename(columns={
              'Win':'Away_Wins',
              'Draw':'Away_Draws',
              'Loss':'Away_Losses'
          })
    )

    team_df = (
        home_counts
        .join(away_counts, how='outer')
        .fillna(0)
        .astype(int)
        .reset_index()
        .rename(columns={'Home':'Club','Away':'Club'})
        .set_index('Club')
    )
    
    return team_df.reset_index()


def weighted_team_results(
    data, 
    alpha = 0.8
):
    """
    Vectorized exponential‐weighted W-D-L sums.
    """
   
    df = data.dropna(subset=['Date','home_score','away_score']).copy()
    
    df = df.sort_values('Date').reset_index(drop=True)
    
    n = len(df)
    
    df['weight'] = alpha ** (n - 1 - np.arange(n))

    home_res = np.select(
        [df.home_score > df.away_score,
         df.home_score == df.away_score],
        ['Win','Draw'],
        default='Loss'
    )
  
    away_res = np.select(
        [df.away_score > df.home_score,
         df.away_score == df.home_score],
        ['Win','Draw'],
        default='Loss'
    )

    home = df[['Home','weight']].assign(Result=home_res, Loc='Home')\
             .rename(columns={'Home':'Club'})
    
    away = df[['Away','weight']].assign(Result=away_res, Loc='Away')\
             .rename(columns={'Away':'Club'})
    
    long = pd.concat([home, away], ignore_index=True)

    weighted = (
        long.pivot_table(
            index='Club',
            columns=['Loc','Result'],
            values='weight',
            aggfunc='sum',
            fill_value=0
        )
    )
  
    weighted.columns = [f"{loc}_{res}_Weighted" for loc,res in weighted.columns]
    
    return weighted.reset_index()


def match_probabilities_dict(
    weighted_df
):
    """
    Build cross‐joined Home vs Away probability lookup.
    """

    home = weighted_df[['Club', 'Home_Win_Weighted', 'Home_Draw_Weighted', 'Home_Loss_Weighted']].copy()
    
    away = weighted_df[['Club', 'Away_Win_Weighted', 'Away_Draw_Weighted', 'Away_Loss_Weighted']].copy()
    
    home.columns = ['Home', 'HW', 'HD', 'HL']
    
    away.columns = ['Away', 'AW', 'AD', 'AL']

    grid = home.merge(away, how='cross').query("Home != Away").reset_index(drop=True)
    
    grid['p_hw'] = np.sqrt(grid['HW'] * grid['AL'])
    
    grid['p_dr'] = np.sqrt(grid['HD'] * grid['AD'])
    
    grid['p_aw'] = np.sqrt(grid['HL'] * grid['AW'])
    
    tot = grid[['p_hw','p_dr','p_aw']].sum(axis=1)
    
    grid[['p_hw','p_dr','p_aw']] = np.where(
        tot.values[:,None] < 1e-12,
        1/3,
        grid[['p_hw','p_dr','p_aw']].div(tot, axis=0)
    )

    return {
        (row['Home'], row['Away']): (row['p_hw'], row['p_dr'], row['p_aw'])
        for _, row in grid.iterrows()
    }

def remaining_matches(
    data
):
    """
    Return unplayed fixtures.
    """
   
    df = data.copy()
    
    return df[df.home_score.isna()][['Date','Home','Away']].reset_index(drop=True)

def expected_league_table(
    team_df, 
    prob_lookup, 
    remain_df
):
    """
    Vectorized expected‐points for each club.
    """
   
    current = (
        team_df
        .set_index('Club')
        .eval('Pts = 3*(Home_Wins+Away_Wins) + (Home_Draws+Away_Draws)')
        ['Pts']
    )

    prob_df = pd.DataFrame([
        {'Home':h,'Away':a,'p_hw':p[0],'p_dr':p[1],'p_aw':p[2]}
        for (h,a), p in prob_lookup.items()
    ])

    rem = remain_df.merge(prob_df, on=['Home','Away'], how='left')
    
    rem[['p_hw', 'p_dr', 'p_aw']] = rem[['p_hw', 'p_dr', 'p_aw']].fillna(1/3)

    rem['home_pts'] = 3 * rem.p_hw + rem.p_dr
    
    rem['away_pts'] = 3 * rem.p_aw + rem.p_dr
 
    pts = pd.concat([
        rem[['Home','home_pts']].rename(columns={'Home':'Club','home_pts':'Pts'}),
        rem[['Away','away_pts']].rename(columns={'Away':'Club','away_pts':'Pts'})
    ]).groupby('Club')['Pts'].sum()

    total = current.add(pts, fill_value=0)
  
    total = total.reindex(team_df['Club']).sort_values(ascending=False)

    return total.reset_index(name='Expected_Points')


def simulate_position_probabilities(
    team_df, 
    remain_df, 
    prob_lookup,
    target_pos, 
    higher = True, 
    n_sims = 10000
):
    """
    Bulk Monte Carlo simulation via NumPy, filtering out blanks first.
    """
  
    clubs = sorted(team_df['Club'])
    
    club_to_idx = {c: i for i, c in enumerate(clubs)}
    
    k = len(clubs)

    base = np.array([
        3 * (row.Home_Wins + row.Away_Wins) + (row.Home_Draws + row.Away_Draws)
        for _, row in team_df.set_index('Club').loc[clubs].iterrows()
    ])

    rm = remain_df.copy()
 
    mask = (
        rm['Home'].isin(clubs) &
        rm['Away'].isin(clubs) &
        (rm['Home'] != '') &
        (rm['Away'] != '')
    )
    
    rm = rm[mask].reset_index(drop=True)

    keys = list(zip(rm.Home, rm.Away))
    
    p = np.array([prob_lookup.get((h,a),(1/3,1/3,1/3)) for h,a in keys])
    
    p_hw, p_dr, p_aw = p[:,0], p[:,1], p[:,2]
    
    M = len(keys)

    home_idx = np.array([club_to_idx[h] for h,a in keys])
    
    away_idx = np.array([club_to_idx[a] for h,a in keys])

    rng = np.random.default_rng(42)
    
    U = rng.random((n_sims, M))
    
    th1 = p_hw[None, :]
    
    th2 = (p_hw + p_dr)[None, :]
    
    outcomes = (U > th1).astype(int) + (U > th2).astype(int)

    home_pts = np.where(outcomes == 0, 3,
               np.where(outcomes == 1, 1, 0))  

    away_pts = np.where(outcomes == 2, 3,
               np.where(outcomes == 1, 1, 0))

    H = np.zeros((k, M))
    
    A = np.zeros((k, M))
    
    H[home_idx, np.arange(M)] = 1
    
    A[away_idx, np.arange(M)] = 1

    sim_pts = H @ home_pts.T + A @ away_pts.T
    
    sim_pts += base[:, None]

    order = np.argsort(-sim_pts, axis=0)
    
    ranks = np.empty_like(order)
    
    ranks[order, np.arange(n_sims)[None, :]] = np.arange(k)[:, None]

    if higher:
    
        counts = np.sum(ranks < target_pos, axis=1)
   
    else:
    
        counts = np.sum(ranks >= target_pos, axis=1)

    probs = counts / n_sims
    
    label = f"P_Finish_{'Top' if higher else 'Below'}_{target_pos}"
    
    return pd.DataFrame({
        'Club': clubs,
        label: probs
    }).sort_values(by=label, ascending=False).reset_index(drop=True)


def build_weighted_df_from_subset(
    played_subset_df, 
    alpha = 0.98
):
    return weighted_team_results(
        data = played_subset_df, 
        alpha = alpha
    )
    

def find_best_alpha(
    data, 
    alphas = np.linspace(0.90, 1.00, 11)
):
    """
    Grid‐search alpha on an 80/20 train/test split.
    """
   
    df = data.dropna(subset=['Date','home_score','away_score']).sort_values('Date')
    
    cutoff = int(0.8 * len(df))
    
    train, test = df.iloc[:cutoff], df.iloc[cutoff:]

    best_alpha, best_acc = None, -1
    
    for alpha in alphas:
    
        wdf = build_weighted_df_from_subset(
            played_subset_df = train, 
            alpha=alpha
        )
        
        prob = match_probabilities_dict(
            weighted_df = wdf
        )
        
        actual = test.apply(lambda r: match_result(
            home_score = r.home_score, 
            away_score = r.away_score
        ), axis=1).values
        
        preds = np.array([
            ["Win", "Draw", "Loss"][int(np.argmax(prob.get((h, a),(1/3, 1/3, 1/3))))]
            for h,a in zip(test.Home, test.Away)
        ])
        
        valid = ~pd.isna(actual)
        
        acc = np.mean(preds[valid] == actual[valid]) if valid.any() else 0
        
        if acc > best_acc:
        
            best_acc, best_alpha = acc, alpha

    print(f"[TRAIN] Best alpha = {best_alpha:.3f}, accuracy = {best_acc:.4f}")
    
    return best_alpha


if __name__ == "__main__":

    league = input("Enter League: ").strip().lower()
    
    position = int(input("Enter Position Number: "))
    
    higher = (input("Higher or Lower than Position? (higher/lower): ").strip().lower() == 'higher')

    data = get_league_data(
        league = league
    )
    
    alpha_star = find_best_alpha(
        data = data, 
        alphas = np.linspace(0.8, 1.0, 10000)
    )

    weighted_df = weighted_team_results(
        data = data, 
        alpha = alpha_star
    )
    
    prob_lookup = match_probabilities_dict(
        weighted_df = weighted_df
    )
    
    team_df = team_results(
        data = data
    )
    
    remain_df = remaining_matches(
        data = data
    )

    exp_table = expected_league_table(
        team_df = team_df, 
        prob_lookup = prob_lookup, 
        remain_df = remain_df
    )
    
    print("----------------------------------------------------")
    print(f"EXPECTED LEAGUE TABLE (Points) with α={alpha_star:.3f}")
    print(exp_table.to_string(index=False))
    print("----------------------------------------------------")

    result_df = simulate_position_probabilities(
        team_df = team_df, 
        remain_df = remain_df, 
        prob_lookup = prob_lookup,
        target_pos = position,
        higher = higher,
        n_sims = 100000
    )
    
    print("POSITION PROBABILITIES:")
    
    print(result_df.to_string(index=False))
