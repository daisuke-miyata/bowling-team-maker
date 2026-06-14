import math
import csv
from collections import defaultdict

import pyomo.environ as pyo


# =========================
# 設定（ここだけ触ればOK）
# =========================

INPUT_CSV = "players2.csv"
OUT_ASSIGNMENT = "assignment.csv"
OUT_TEAMS = "teams.csv"

TEAM_NO_START = 1
MAX_TEAMS = 10           # ★追加：最大チーム数（=レーン数）
MAX_PLAYERS = 4 * MAX_TEAMS  # ★追加：最大人数（5人チーム禁止なので40）

LAMBDA_GENDER = 20.0
SOLVER = "cbc"
TIME_LIMIT_SEC = 10
TEE = False
CBC_MIP_GAP = 0.01
USE_SYMMETRY_BREAK = True


# =========================
# CSV読み込み：name,gender,score
# gender: M/F
# =========================

def read_players_from_csv(path):
    names, scores, genders = {}, {}, {}  # gender: F=1, M=0
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        req = {"name", "gender", "score"}
        if r.fieldnames is None or not req.issubset(set(r.fieldnames)):
            raise ValueError(f"CSVヘッダ不足。必要: {req}, 実際: {r.fieldnames}")

        idx = 0
        for row in r:
            name = row["name"].strip()
            if not name:
                continue
            gender = row["gender"].strip().upper()
            if gender not in ("M", "F"):
                raise ValueError(f"gender は M/F を想定: {row}")
            try:
                score = int(float(row["score"]))
            except Exception:
                raise ValueError(f"score が数値でない: {row}")

            names[idx] = name
            scores[idx] = score
            genders[idx] = 1 if gender == "F" else 0
            idx += 1

    if idx == 0:
        raise ValueError("CSVに参加者がいません。")
    return names, scores, genders


# =========================
# solver別：時間制限などのオプション
# =========================

def set_solver_options(solver, solver_name, time_limit_sec):
    s = solver_name.lower()

    if time_limit_sec is not None:
        if s == "cbc":
            solver.options["seconds"] = int(time_limit_sec)
        elif s == "glpk":
            solver.options["tmlim"] = int(time_limit_sec)
        elif s == "highs":
            solver.options["time_limit"] = float(time_limit_sec)
        elif s == "gurobi":
            solver.options["TimeLimit"] = float(time_limit_sec)
        elif s == "cplex":
            solver.options["timelimit"] = float(time_limit_sec)
        else:
            solver.options["timelimit"] = float(time_limit_sec)

    if s == "cbc" and CBC_MIP_GAP is not None:
        solver.options["ratioGap"] = float(CBC_MIP_GAP)


# =========================
# 指定チーム数Tで解く（3/4人のみ、5人禁止）
# =========================

def solve_for_T(names, scores, genders, T,
                lambda_gender=LAMBDA_GENDER,
                solver_name=SOLVER,
                time_limit_sec=TIME_LIMIT_SEC):
    I = list(scores.keys())
    N = len(I)

    # 3/4人チームだけで成立するTの範囲
    if not (math.ceil(N / 4) <= T <= math.floor(N / 3)):
        return None

    # 4人チーム数は N と T から一意：sum(z)=N-3T
    z_sum = N - 3 * T
    if not (0 <= z_sum <= T):
        return None

    mu = sum(scores[i] for i in I) / N
    p = sum(genders[i] for i in I) / N

    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=I)
    m.Teams = pyo.Set(initialize=list(range(T)))

    m.x = pyo.Var(m.I, m.Teams, domain=pyo.Binary)
    m.z = pyo.Var(m.Teams, domain=pyo.Binary)           # 0:3人 1:4人
    m.S = pyo.Var(m.Teams, domain=pyo.NonNegativeReals) # 合計点
    m.F = pyo.Var(m.Teams, domain=pyo.NonNegativeReals) # 女性数
    m.D = pyo.Var(domain=pyo.NonNegativeReals)          # 合計点偏差(max)
    m.G = pyo.Var(domain=pyo.NonNegativeReals)          # 女性数偏差(max)

    m.one_team = pyo.Constraint(m.I, rule=lambda m, i: sum(m.x[i, t] for t in m.Teams) == 1)
    m.team_size = pyo.Constraint(m.Teams, rule=lambda m, t: sum(m.x[i, t] for i in m.I) == 3 + m.z[t])
    m.z_count = pyo.Constraint(expr=sum(m.z[t] for t in m.Teams) == z_sum)

    m.score_def = pyo.Constraint(m.Teams, rule=lambda m, t: m.S[t] == sum(scores[i] * m.x[i, t] for i in m.I))
    m.female_def = pyo.Constraint(m.Teams, rule=lambda m, t: m.F[t] == sum(genders[i] * m.x[i, t] for i in m.I))

    m.score_dev_pos = pyo.Constraint(m.Teams, rule=lambda m, t: (m.S[t] - mu * (3 + m.z[t])) <= m.D)
    m.score_dev_neg = pyo.Constraint(m.Teams, rule=lambda m, t: -(m.S[t] - mu * (3 + m.z[t])) <= m.D)
    m.gender_dev_pos = pyo.Constraint(m.Teams, rule=lambda m, t: (m.F[t] - p * (3 + m.z[t])) <= m.G)
    m.gender_dev_neg = pyo.Constraint(m.Teams, rule=lambda m, t: -(m.F[t] - p * (3 + m.z[t])) <= m.G)

    # 対称性破壊（速くする）
    if USE_SYMMETRY_BREAK:
        top = sorted(I, key=lambda i: scores[i], reverse=True)[:T]
        m.seed = pyo.ConstraintList()
        for t, i in enumerate(top):
            m.seed.add(m.x[i, t] == 1)

        m.z_order = pyo.ConstraintList()
        for t in range(T - 1):
            m.z_order.add(m.z[t] >= m.z[t + 1])

    m.obj = pyo.Objective(expr=m.D + lambda_gender * m.G, sense=pyo.minimize)

    solver = pyo.SolverFactory(solver_name)
    if solver is None or not solver.available():
        raise RuntimeError(f"Solver '{solver_name}' が使えません。cbc/glpk/highs を用意してください。")
    set_solver_options(solver, solver_name, time_limit_sec)
    res = solver.solve(m, tee=TEE)

    assign, teams = {}, defaultdict(list)
    for i in I:
        for t in range(T):
            v = pyo.value(m.x[i, t])
            if v is not None and v > 0.5:
                assign[i] = t
                teams[t].append(i)
                break
        else:
            return None  # 解が取れてない（タイムアウト等）

    summary = []
    for t in range(T):
        mem = teams[t]
        size = len(mem)
        tot = sum(scores[i] for i in mem)
        avg = tot / size if size else 0.0
        fem = sum(genders[i] for i in mem)
        summary.append((t, size, tot, avg, fem))

    return {
        "T": T, "assign": assign, "teams": teams, "summary": summary,
        "mu": mu, "p": p,
        "D": float(pyo.value(m.D)), "G": float(pyo.value(m.G)),
        "obj": float(pyo.value(m.obj)),
        "solver_result": res
    }


# =========================
# 人数ぶれ対応：成立するTを総当たりして最良を選ぶ
# ★最大チーム数(MAX_TEAMS)を超えない + 40人超えはエラーで停止
# =========================

def solve_auto_T(names, scores, genders,
                 lambda_gender=LAMBDA_GENDER,
                 solver_name=SOLVER,
                 time_limit_sec=TIME_LIMIT_SEC,
                 max_teams=MAX_TEAMS):
    N = len(scores)

    # ★ここが運営制約：40人超えを明確に止める
    if N > 4 * max_teams:
        raise ValueError(
            f"参加者が {N} 人のため、最大 {max_teams} チーム（=最大 {4*max_teams} 人）では運営できません。"
        )

    Tmin = math.ceil(N / 4)
    Tmax = min(math.floor(N / 3), max_teams)

    if Tmin > Tmax:
        raise ValueError(
            f"N={N} は 3/4人チームのみ & 最大{max_teams}チームでは構成できません。"
            f"（必要Tは {Tmin} 以上）"
        )

    best = None
    for T in range(Tmin, Tmax + 1):
        r = solve_for_T(names, scores, genders, T, lambda_gender, solver_name, time_limit_sec)
        if r is None:
            continue
        if best is None or (r["obj"], r["D"], r["G"], r["T"]) < (best["obj"], best["D"], best["G"], best["T"]):
            best = r
    return best


# =========================
# CSV出力
# =========================

def export_assignment_csv(path, names, scores, genders, assign, team_offset=TEAM_NO_START):
    rows = [{
        "name": names[i],
        "gender": "F" if genders[i] else "M",
        "score": scores[i],
        "team": assign[i] + team_offset
    } for i in sorted(names.keys(), key=lambda k: names[k])]

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "gender", "score", "team"])
        w.writeheader()
        w.writerows(rows)


def export_teams_csv(path, names, scores, genders, teams, summary, team_offset=TEAM_NO_START):
    rows = []
    for (t, size, tot, avg, fem) in sorted(summary, key=lambda x: x[0]):
        mem = sorted(teams[t], key=lambda i: scores[i], reverse=True)
        mems = "; ".join([f"{names[i]}({'F' if genders[i] else 'M'}:{scores[i]})" for i in mem])
        rows.append({
            "team": t + team_offset, "team_size": size, "total": tot,
            "avg": round(avg, 2), "females": fem, "members": mems
        })

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["team", "team_size", "total", "avg", "females", "members"])
        w.writeheader()
        w.writerows(rows)


def main():
    names, scores, genders = read_players_from_csv(INPUT_CSV)

    N = len(scores)
    M = sum(1 for i in genders if genders[i] == 0)
    F = sum(1 for i in genders if genders[i] == 1)
    print(f"Participants N={N} (M={M}, F={F})")
    print("Allowed team sizes: 3 or 4 only (no 5)")
    print(f"Max teams (lanes): {MAX_TEAMS}  -> Max players: {MAX_PLAYERS}\n")

    result = solve_auto_T(names, scores, genders)

    if result is None:
        raise RuntimeError("解が見つかりませんでした（タイムアウト or 制約が厳しすぎる可能性）。")

    print(f"=== Selected T={result['T']} ===")
    print(f"mu={result['mu']:.2f} | female ratio(p)={result['p']:.2f}")
    print(f"Max score deviation D={result['D']:.2f} | Max gender deviation G={result['G']:.2f} | obj={result['obj']:.2f}\n")

    by_team = defaultdict(list)
    for i, t in result["assign"].items():
        by_team[t].append(i)

    for t in range(result["T"]):
        mem = sorted(by_team[t], key=lambda i: scores[i], reverse=True)
        print(f"[Team {t + TEAM_NO_START}]")
        for i in mem:
            sex = "F" if genders[i] else "M"
            print(f"  {names[i]:<12}  {sex}  score={scores[i]:>3}")
        print()

    export_assignment_csv(OUT_ASSIGNMENT, names, scores, genders, result["assign"])
    export_teams_csv(OUT_TEAMS, names, scores, genders, result["teams"], result["summary"])
    print(f"Saved: {OUT_ASSIGNMENT}")
    print(f"Saved: {OUT_TEAMS}")


if __name__ == "__main__":
    main()
