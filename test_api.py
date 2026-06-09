from app import app

client = app.test_client()

print('GET /health')
print(client.get('/health').json)

print('\nGET /api/leaderboard?limit=3')
print(client.get('/api/leaderboard?limit=3').json)

print('\nPOST /api/predict')
payload = {
    'runs_off_bat': 4,
    'extras': 0,
    'current_run_rate': 8.5,
    'required_run_rate': 9.2,
    'pressure_index': 0.72,
    'expected_runs': 1.4,
    'expected_wicket_probability': 0.08,
    'shot_risk_score': 0.45,
    'batter_control_score': 0.82,
    'target_is_wicket': 0,
    'target_total_runs': 4,
}
print(client.post('/api/predict', json=payload).json)
