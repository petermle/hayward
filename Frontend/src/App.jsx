import "./App.css"
import { useState } from "react"
import SearchPage from "./SearchPage"

export default function App() {
	const [page, setPage] = useState("home")
	const [query, setQuery] = useState("")

	return (
		<div className="container">
			<div className="nav">
				<button className="login" onClick={() => setPage("login")}>
					Log In
				</button>
				<button className="signup" onClick={() => setPage("signup")}>
					Sign up
				</button>
			</div>

			{/* HOME */}
			{page === "home" && (
				<div className="section">
					<div className="center">
						<div className="logo">WoW</div>

						<h1>Enter your school to get started</h1>

						<input
							className="input"
							placeholder="Your School"
							value={query}
							onChange={(e) => setQuery(e.target.value)}
							onKeyDown={(e) => {
								if (e.key === "Enter" && query.trim() !== "") {
									setPage("search")
								}
							}}
						/>
					</div>
				</div>
			)}

			{/* SEARCH */}
			{page === "search" && (
				<SearchPage query={query} />
			)}

			{/* LOGIN */}
			{page === "login" && (
				<div className="section">
					<div className="center">
						<div className="logo">WoW</div>
						<h1>Log In</h1>
						<input className="input" placeholder="Email" />
						<input className="input" placeholder="Password" type="password" />
					</div>
				</div>
			)}

			{/* SIGNUP */}
			{page === "signup" && (
				<div className="section">
					<div className="center">
						<div className="logo">WoW</div>
						<h1>Sign Up</h1>
						<input className="input" placeholder="Email" />
						<input className="input" placeholder="Password" type="password" />
					</div>
				</div>
			)}
		</div>
	)
}