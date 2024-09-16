<script>
	import { onMount } from 'svelte';
	import { marked } from 'marked';
  
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
			model: 'faragia-dev',
			messages: [
			  { role: 'user', content: question }
			]
		  }),
		});
		
		if (!response.ok) {
		  throw new Error('API request failed');
		}
		
		const data = await response.json();
		answer = data.choices[0].message.content;
	  } catch (e) {
		error = 'An error occurred while fetching the answer.';
		console.error(e);
	  } finally {
		loading = false;
	  }
	}
  
	function formatAnswer(text) {
	  // Replace newlines with <br> tags
	  const htmlString = text.replace(/\n/g, '<br>');
	  // Parse the resulting HTML string as markdown
	  return marked(htmlString);
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
	  <div class="answer">{@html formatAnswer(answer)}</div>
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
	.answer {
	  background-color: #f4f4f4;
	  padding: 10px;
	  border-radius: 5px;
	  white-space: pre-wrap;
	  word-wrap: break-word;
	}
	.error {
	  color: red;
	}
	:global(.answer h1, .answer h2, .answer h3, .answer h4, .answer h5, .answer h6) {
	  margin-top: 1em;
	  margin-bottom: 0.5em;
	}
	:global(.answer p) {
	  margin-bottom: 1em;
	}
	:global(.answer ul, .answer ol) {
	  margin-bottom: 1em;
	  padding-left: 2em;
	}
	:global(.answer pre) {
	  background-color: #e0e0e0;
	  padding: 10px;
	  border-radius: 3px;
	  overflow-x: auto;
	}
	:global(.answer code) {
	  font-family: monospace;
	  background-color: #e0e0e0;
	  padding: 2px 4px;
	  border-radius: 3px;
	}
  </style>