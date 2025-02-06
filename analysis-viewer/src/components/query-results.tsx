import * as React from 'react'
import { AlertCircle, CheckCircle2, Code2, PlayCircle, Terminal } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';

// Type definitions for RAG output
interface ExecutionStep {
  name: string;
  success: boolean;
  output?: string;
  error?: string;
  status?: string;
}

interface ExecutionResults {
  success: boolean;
  output?: string;
  error?: string;
  results?: Record<string, string | number | boolean | null | Array<unknown>>;
}

interface SetupResults {
  success: boolean;
  error?: string;
}

interface AnalysisResults {
  explanation?: string;
  setup?: SetupResults;
  steps?: ExecutionStep[];
  execution?: ExecutionResults;
}

const QueryResults = ({ results }: { results: AnalysisResults }) => {
  if (!results) {
    return (
      <Alert>
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>No Results</AlertTitle>
        <AlertDescription>No analysis results available.</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-4 w-full max-w-4xl">
      {/* Explanation Card */}
      {results.explanation && (
        <Card>
          <CardHeader>
            <CardTitle>Analysis Explanation</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-gray-700">{results.explanation}</p>
          </CardContent>
        </Card>
      )}

      {/* Setup Results */}
      {results.setup && (
        <Card>
          <CardHeader className="flex flex-row items-center gap-2">
            <Terminal className="h-5 w-5" />
            <CardTitle>Setup</CardTitle>
          </CardHeader>
          <CardContent>
            {results.setup.success ? (
              <Alert className="bg-green-50">
                <CheckCircle2 className="h-4 w-4" />
                <AlertTitle>Setup completed successfully</AlertTitle>
              </Alert>
            ) : (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Setup failed</AlertTitle>
                {results.setup.error && (
                  <AlertDescription>{results.setup.error}</AlertDescription>
                )}
              </Alert>
            )}
          </CardContent>
        </Card>
      )}

      {/* Function Definition Steps */}
      {results.steps && results.steps.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Code2 className="h-5 w-5" />
              <CardTitle>Function Definitions</CardTitle>
            </div>
            <CardDescription>Results of defining analysis functions</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {results.steps.map((step, index) => (
                <div key={index} className="border rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2">
                    {step.success ? (
                      <CheckCircle2 className="h-4 w-4 text-green-500" />
                    ) : (
                      <AlertCircle className="h-4 w-4 text-red-500" />
                    )}
                    <h3 className="font-medium">{step.name || `Step ${index + 1}`}</h3>
                  </div>
                  
                  {step.output && (
                    <pre className="bg-gray-50 p-3 rounded-md text-sm mt-2 overflow-x-auto whitespace-pre-wrap">
                      {step.output}
                    </pre>
                  )}
                  
                  {!step.success && step.error && (
                    <Alert variant="destructive" className="mt-2">
                      <AlertCircle className="h-4 w-4" />
                      <AlertTitle>Error</AlertTitle>
                      <AlertDescription>{step.error}</AlertDescription>
                    </Alert>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Execution Results */}
      {results.execution && (
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <PlayCircle className="h-5 w-5" />
              <CardTitle>Execution Results</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            {results.execution.success ? (
              <div className="space-y-4">
                {results.execution.output && (
                  <pre className="bg-gray-50 p-4 rounded-md text-sm text-black">
                    {results.execution.output}
                  </pre>
                )}
                {results.execution.results && (
                  <div className="border rounded-lg p-4 bg-white">
                    <h4 className="font-medium mb-2 text-lg text-black">Results:</h4>
                    <div className="grid grid-cols-1 gap-4">
                      {Object.entries(results.execution.results).map(([key, value]) => (
                        <div key={key} className="bg-gray-50 p-4 rounded-lg">
                          <span className="font-mono font-medium text-black">{key}:</span>
                          <pre className="mt-2 text-sm text-gray-700 whitespace-pre-wrap overflow-x-auto">
                            {typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}
                          </pre>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Execution failed</AlertTitle>
                <AlertDescription>{results.execution.error}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default QueryResults;