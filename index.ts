import { Configuration, OpenAIApi } from "openai"
import tags from "./tags"

const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
})
const openai = new OpenAIApi(configuration)

async function init() {
  try {
    const article = await Bun.file("article.txt").text()
    const articleWords = article
      .split(" ")
      .map((i) => i.trim())
      .slice(0, 100)
    const response = await openai.createEmbedding({
      model: "text-embedding-ada-002",
      input: [...articleWords, ...tags],
    })
    const embeddings = response.data.data.map(
      (embedding) => embedding.embedding
    )
    const articleWordEmbeddings = embeddings.slice(0, articleWords.length)
    const tagEmbeddings = embeddings.slice(articleWords.length)

    let matchedTags: { similarity: number; tag: string }[] = []
    articleWordEmbeddings.forEach((articleWordEmbedding, i) => {
      tagEmbeddings.forEach((tagEmbedding, j) => {
        const similarity = cosineSimilarity(articleWordEmbedding, tagEmbedding)
        if (similarity > 0.87 && !matchedTags.some((i) => i.tag === tags[j])) {
          matchedTags.push({
            similarity,
            tag: tags[j],
          })
        }
      })
    })

    console.log(
      "Matched Tags:",
      matchedTags.sort((a, b) => b.similarity - a.similarity).map((i) => i.tag)
    )
  } catch (e) {
    console.error("Failed to generate embeddings", e, e.response.data)
  }
}

function cosineSimilarity(a: number[], b: number[]) {
  const dotProduct = a.reduce((acc, val, i) => acc + val * b[i], 0)
  const magnitudeA = Math.sqrt(a.reduce((acc, val) => acc + val * val, 0))
  const magnitudeB = Math.sqrt(b.reduce((acc, val) => acc + val * val, 0))
  return dotProduct / (magnitudeA * magnitudeB)
}

init()
