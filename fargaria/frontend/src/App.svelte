<script>
	import { onMount } from 'svelte';
  
	let question = '';
	let answer = '';
	let loading = false;
	let error = '';
  
	async function handleSubmit() {
	  loading = true;
	  error = '';
	  try {
		const response = await fetch('/v1/chat/completions', {
		  method: 'POST',
		  headers: {
			'Content-Type': 'application/json',
		  },
		  body: JSON.stringify({
			model: 'gpt-4',  // You might want to make this configurable
			messages: [
			  { role: 'user', content: question }
			]
		  }),
		});
		
		if (!response.ok) {
		  throw new Error('API request failed');
		}
		
		const data = await response.json();
		answer = JSON.parse(data.choices[0].message.content);
	  } catch (e) {
		error = 'An error occurred while fetching the answer.';
		console.error(e);
	  } finally {
		loading = false;
	  }
	}
  </script>
  
  <main>
	<h1>Fragaria: Chain of Thought Reasoning</h1>
	
	<form on:submit|preventDefault={handleSubmit}>
	  <label for="question">Enter your question:</label>
	  <textarea id="question" bind:value={question} rows="4" cols="50"></textarea>
	  <button type="submit" disabled={loading}>Submit</button>
	</form>
  
	{#if loading}
	  <p>Thinking...</p>
	{:else if error}
	  <p class="error">{error}</p>
	{:else if answer}
	  <h2>Answer:</h2>
	  <pre>{JSON.stringify(answer, null, 2)}</pre>
	{/if}
  </main>
  
  <style>
	main {
	  max-width: 800px;
	  margin: 0 auto;
	  padding: 20px;
	  font-family: Arial, sans-serif;
	}
	form {
	  display: flex;
	  flex-direction: column;
	  gap: 10px;
	  margin-bottom: 20px;
	}
	textarea {
	  width: 100%;
	  padding: 10px;
	}
	button {
	  padding: 10px;
	  background-color: #4CAF50;
	  color: white;
	  border: none;
	  cursor: pointer;
	}
	button:disabled {
	  background-color: #cccccc;
	}
	pre {
	  background-color: #f4f4f4;
	  padding: 10px;
	  border-radius: 5px;
	  white-space: pre-wrap;
	  word-wrap: break-word;
	}
	.error {
	  color: red;
	}
  </style>