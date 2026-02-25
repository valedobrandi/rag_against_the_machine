#!/bin/sh
jq -s '
  . as [$docs, $results]
  | [range(0; 50) | {
      index: .,
      question: $docs.rag_questions[.].question,
      expected: $docs.rag_questions[.].answer,
      predicted: $results.search_results[.].answer
    }]
' data/datasets/AnsweredQuestions/dataset_docs_public.json data/output/search_results_and_answer/dataset_docs_public.json

echo "Calculating IDK percentage..."
jq -s '
  .[0].search_results | {
    total_questions: length,
    idk_count: (map(select(.answer | test("know"; "i"))) | length)
  } | . + { idk_percentage: ((.idk_count / .total_questions) * 100 | round) }
' data/output/search_results_and_answer/dataset_docs_public.json