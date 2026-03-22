import requests
from bs4 import BeautifulSoup
import time

headers = {
	"User-Agent": "Mozilla/5.0"
}

# 🔥 YOUR SUPER DB ENDPOINTS
BASE_URL = "https://your-api.com/people"


# ---------------- DB CHECK ----------------
def db_get(name, school, target_type):
	params = {
		"name": name,
		"school": school,
		"type": target_type
	}

	res = requests.get(BASE_URL, params=params)

	if res.status_code == 200:
		data = res.json()
		if data:
			return data

	return None


def db_insert(data):
	res = requests.post(BASE_URL, json=data)
	return res.status_code == 200


# ---------------- SEARCH ----------------
def search_linkedin(name, school):
	query = f'site:linkedin.com/in "{name}" "{school}"'
	url = f"https://www.google.com/search?q={query.replace(' ', '+')}"

	res = requests.get(url, headers=headers)
	soup = BeautifulSoup(res.text, "html.parser")

	results = []

	for g in soup.select("div.tF2Cxc"):
		a = g.find("a")
		h3 = g.find("h3")

		if a and h3:
			results.append({
				"title": h3.text,
				"link": a["href"],
				"text": h3.text
			})

	# fallback
	if not results:
		for a in soup.find_all("a"):
			href = a.get("href")
			if href and "linkedin.com/in" in href:
				results.append({
					"title": a.get_text(),
					"link": href,
					"text": a.get_text()
				})

	return results


# ---------------- SCORING ----------------
def score_result(result, name, school, target_type):
	text = (result["title"] + " " + result["text"]).lower()
	score = 0

	if name.lower() in text:
		score += 5

	for part in name.lower().split():
		if part in text:
			score += 2

	if school.lower() in text:
		score += 4

	prof_keywords = [
		"professor", "assistant professor",
		"associate professor", "lecturer"
	]

	if target_type == "professor":
		if any(k in text for k in prof_keywords):
			score += 6

	if target_type == "student":
		if "student" in text:
			score += 3

	return score


def pick_best(results, name, school, target_type):
	scored = [(score_result(r, name, school, target_type), r) for r in results]
	scored.sort(reverse=True, key=lambda x: x[0])

	if not scored:
		return None

	best_score, best = scored[0]

	if best_score < 2:
		return None

	return best


# ---------------- MAIN PIPELINE ----------------
def get_person(name, school, target_type="student"):
	# 1. CHECK SUPER DB
	cached = db_get(name, school, target_type)

	if cached:
		print("SUPER DB HIT")
		return cached

	print("NOT IN DB, FETCHING...")

	# 2. SEARCH
	results = search_linkedin(name, school)

	if not results:
		return None

	# 3. MATCH
	best = pick_best(results, name, school, target_type)

	if not best:
		return None

	data = {
		"name": name,
		"school": school,
		"type": target_type,
		"link": best["link"]
	}

	# 4. STORE IN SUPER DB
	db_insert(data)

	return data


# ---------------- TEST ----------------
if __name__ == "__main__":
	people = [
		("Andrew Ng", "Stanford University", "professor"),
		("Angel Cruz", "San Jose State University", "student")
	]

	for name, school, t in people:
		result = get_person(name, school, t)
		print(result)
		time.sleep(2)