'use client';

import { useState } from 'react';
import QueryResults from '@/components/query-results';
import Papa from 'papaparse';

export default function Home() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [csvData, setCsvData] = useState(null);
  const [fileName, setFileName] = useState('');
  const [columns, setColumns] = useState([]);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setFileName(file.name);
      Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (results) => {
          if (results.errors.length > 0) {
            alert(`Error parsing CSV: ${results.errors[0].message}`);
            return;
          }
          setCsvData(results.data);
          setColumns(Object.keys(results.data[0] || {}));
        },
        error: (error) => {
          alert(`Error parsing CSV: ${error.message}`);
        }
      });
    }
  };

  const handleAnalyze = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query,
          input_data: {
            df: csvData || [],
            schema: columns
          }
        }),
      });
      
      if (!response.ok) {
        throw new Error('Analysis failed');
      }
      
      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error('Error:', error);
      setResults({
        explanation: 'Error performing analysis',
        setup: { success: false, error: error.message }
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-8">
      <h1 className="text-3xl font-bold mb-8">Analysis Results</h1>
      
      <div className="w-full max-w-4xl mb-8 space-y-4">
        <div className="border rounded-lg p-4">
          <h2 className="text-xl font-semibold mb-2">1. Upload Data</h2>
          <div className="flex flex-col space-y-2">
            <input
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              className="block w-full text-sm text-gray-500
                file:mr-4 file:py-2 file:px-4
                file:rounded-md file:border-0
                file:text-sm file:font-semibold
                file:bg-blue-50 file:text-blue-700
                hover:file:bg-blue-100"
            />
            {fileName && (
              <div className="text-sm text-gray-600">
                <p>Uploaded file: {fileName}</p>
                {columns.length > 0 && (
                  <div className="mt-2">
                    <p className="font-medium">Available columns:</p>
                    <p className="text-gray-500">{columns.join(', ')}</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        <div className="border rounded-lg p-4">
          <h2 className="text-xl font-semibold mb-2">2. Enter Query</h2>
          <textarea
            className="w-full p-4 border rounded-lg mb-4 text-black"
            rows={4}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your analysis query here..."
          />
          <button
            onClick={handleAnalyze}
            disabled={loading || !csvData}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-400"
          >
            {loading ? 'Analyzing...' : 'Analyze'}
          </button>
          {!csvData && (
            <p className="text-sm text-gray-600 mt-2">
              Please upload a CSV file before analyzing
            </p>
          )}
        </div>
      </div>

      {results && <QueryResults results={results} />}
    </main>
  );
}