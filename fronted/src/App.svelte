<script>
  import { invoke } from '@tauri-apps/api/tauri';
  import { onMount } from 'svelte';
  import Prism from 'prismjs';
  import 'prismjs/themes/prism-dark.css';
  import 'prismjs/components/prism-python';
  import 'prismjs/components/prism-javascript';

  let prompt = '';
  let result = '';
  let imageUrl = '';
  let history = [];
  let loading = false;
  let mode = 'text'; // 'text' lub 'image'

  async function sendRequest() {
    if (!prompt.trim()) return;
    loading = true;
    result = '';
    imageUrl = '';

    try {
      const response = await invoke('invoke_python', { action: mode, prompt });
      const data = JSON.parse(response);
      if (data.status === 'success') {
        if (mode === 'image') {
          imageUrl = `file://${data.data}`;
        } else {
          result = data.data;
          setTimeout(() => Prism.highlightAll(), 0);
        }
      } else {
        result = data.data;
      }
    } catch (error) {
      result = `Błąd: ${error}`;
    }
    loading = false;
    fetchHistory();
  }

  async function fetchHistory() {
    try {
      const response = await invoke('invoke_python', { action: 'history', prompt: '' });
      history = JSON.parse(response).data;
    } catch (error) {
      result = `Błąd pobierania historii: ${error}`;
    }
  }

  function handleKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendRequest();
    }
  }

  onMount(() => {
    fetchHistory();
    Prism.highlightAll();
  });
</script>

<style>
  .container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    font-family: Arial, sans-serif;
  }

  textarea {
    width: 100%;
    height: 100px;
    margin-bottom: 10px;
    resize: vertical;
  }

  button {
    padding: 10px 20px;
    margin-right: 10px;
    cursor: pointer;
  }

  .history {
    max-height: 300px;
    overflow-y: auto;
    margin-bottom: 20px;
    border: 1px solid #ccc;
    padding: 10px;
  }

  .message {
    margin: 10px 0;
  }

  .loading {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }

  img {
    max-width: 100%;
    margin-top: 20px;
  }

  pre {
    background: #1e1e1e;
    padding: 10px;
    border-radius: 5px;
  }
</style>

<div class="container">
  <h1>Hex AI</h1>
  
  <div class="history">
    {#each history as msg}
      <div class="message">
        <strong>{msg.action === 'image' ? 'Obraz' : 'Tekst'} ({msg.timestamp}):</strong>
        <p>Prompt: {msg.prompt}</p>
        {#if msg.action === 'image'}
          <img src="file://{msg.response}" alt="Generated image" />
        {:else}
          <pre><code class="language-{msg.prompt.includes('python') ? 'python' : 'javascript'}">{msg.response}</code></pre>
        {/if}
      </div>
    {/each}
  </div>

  <select bind:value={mode}>
    <option value="text">Tekst/Kod</option>
    <option value="image">Obraz</option>
  </select>

  <textarea bind:value={prompt} placeholder="Wpisz zapytanie..." on:keydown={handleKeydown}></textarea>
  <button on:click={sendRequest} disabled={loading}>
    {#if loading}
      <span class="loading">⏳</span> Wczytywanie...
    {:else}
      Generuj
    {/if}
  </button>

  {#if result}
    <pre><code class="language-{prompt.includes('python') ? 'python' : 'javascript'}">{result}</code></pre>
  {/if}
  {#if imageUrl}
    <img src={imageUrl} alt="Generated image" />
  {/if}
</div>
