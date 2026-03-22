import { useEffect, useState } from "react"
import "./SearchPage.css"

export default function SearchPage({ query }) {
	const [results, setResults] = useState([])

	useEffect(() => {
		fetch(`http://127.0.0.1:8000/search?name=${query}`)
			.then(res => res.json())
			.then(data => setResults(data))
	}, [query])

	return (
		<div className="search-page">
			<h2>{results.length} professors for "{query}"</h2>

			<div className="results">
				{results.map((prof) => (
					<div className="card" key={prof.id}>
						<div className="rating-box">
							{prof.rating}
						</div>

						<div className="school-name">
							{prof.name}
							<p>{prof.department}</p>
						</div>

						<div className="location">
							{prof.numRatings} ratings
						</div>
					</div>
				))}
			</div>
		</div>
	)
}