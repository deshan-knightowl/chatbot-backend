import express from "express";
import pkg from "body-parser";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const { json } = pkg;
app.use(json());

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const index = pinecone.index(process.env.PINECONE_INDEX_NAME);

function normalizeText(text) {
  return text.trim().toLowerCase(); // case-insensitive match
}

// --- EMBED ENDPOINT ---
app.post("/embed", async (req, res) => {
  const texts = req.body;

  if (
    !Array.isArray(texts) ||
    !texts.every((item) => item.text && typeof item.text === "string")
  ) {
    return res.status(400).json({
      error:
        "Invalid input format, expected an array of objects with a 'text' field.",
    });
  }

  try {
    const embeddings = [];

    for (const { text } of texts) {
      const normalizedText = normalizeText(text);

      const response = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: normalizedText,
        dimensions: 512,
      });

      const embedding = response.data[0].embedding;

      await index.upsert([
        {
          id: `text-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          values: embedding,
          metadata: { text: normalizedText }, // store normalized
        },
      ]);

      embeddings.push({ text: normalizedText, embedding });
    }

    res.status(200).send({ message: "Texts embedded successfully", embeddings });
  } catch (error) {
    console.error("Error in /embed:", error);
    res.status(500).send({ error: error.message });
  }
});

// --- CHAT ENDPOINT ---
app.post("/chat", async (req, res) => {
  const { query } = req.body;

  if (!query || typeof query !== "string") {
    return res.status(400).json({ error: "Invalid query input" });
  }

  try {
    const normalizedQuery = normalizeText(query);

    const embeddingResponse = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: [normalizedQuery],
      dimensions: 512,
    });

    const embedding = embeddingResponse.data[0].embedding;

    const pineconeResponse = await index.query({
      vector: embedding,
      topK: 5,
      includeValues: false,
      includeMetadata: true,
    });

    const contextMatches = pineconeResponse.matches
      .filter((match) => match.score > 0.7) // Filter by match threshold
      .map((match) => match.metadata?.text)
      .filter(Boolean);

    const context = contextMatches.join("\n");

    const basePrompt = `Context:\n${context || "None found"}\n\nQuestion: ${query}`;

    const messages = [
      {
        role: "system",
        content:
          "You are the AI chatbot of this website. Only answer based on the provided company knowledge and context. If unsure, still try to help using any relevant context retrieved.",
      },
      {
        role: "user",
        content: basePrompt,
      },
    ];

    const chatResponse = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages,
    });

    const reply = chatResponse.choices?.[0]?.message?.content?.trim();

    res.status(200).json({
      response: reply || "Sorry, no appropriate response could be generated.",
    });
  } catch (error) {
    console.error("Error in /chat:", error);
    res.status(500).json({
      error: error.message || "An error occurred while processing your request",
    });
  }
});

app.get("/", (req, res) => {
  res.send("Hello, World!");
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
