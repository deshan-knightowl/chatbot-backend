import express from "express";
import pkg from "body-parser";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";
import dotenv from "dotenv";
import { randomUUID } from "crypto";

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

// Embed and store texts
app.post("/embed", async (req, res) => {
  console.log("Request received at /embed");

  const texts = req.body;

  if (
    !Array.isArray(texts) ||
    !texts.every((item) => item.text && typeof item.text === "string")
  ) {
    return res.status(400).json({
      error: "Invalid input: expected an array of objects with a 'text' field.",
    });
  }

  try {
    const inputs = texts.map((item) => item.text);
    const response = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: inputs,
      dimensions: 512,
    });

    const upserts = response.data.map((embeddingObj, i) => ({
      id: `text-${randomUUID()}`,
      values: embeddingObj.embedding,
      metadata: { text: inputs[i] },
    }));

    await index.upsert(upserts);

    const result = upserts.map(({ metadata, values }) => ({
      text: metadata.text,
      embedding: values,
    }));

    res.status(200).send({ message: "Texts embedded successfully", embeddings: result });
  } catch (error) {
    console.error("Error in /embed:", error);
    res.status(500).send({ error: error.message });
  }
});

// Chat endpoint with retrieval from Pinecone
app.post("/chat", async (req, res) => {
  const { query } = req.body;

  if (!query || typeof query !== "string") {
    return res.status(400).json({ error: "Invalid query input" });
  }

  try {
    const embeddingResponse = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: [query],
      dimensions: 512,
    });

    const queryEmbedding = embeddingResponse.data[0].embedding;

    const pineconeResponse = await index.query({
      vector: queryEmbedding,
      topK: 8,
      includeMetadata: true,
    });

    const matches = pineconeResponse.matches || [];

    const context = matches
      .filter((match) => match.score > 0.4) // threshold for meaningful matches
      .map((match) => `- ${match.metadata?.text}`)
      .join("\n");

    const fallback = matches.length > 0 ? matches[0].metadata?.text : null;

    const systemPrompt =
      "You are an assistant chatbot trained on this website's content. Answer questions using the given context. If you're unsure, respond with: 'I'm only trained to answer questions related to this website.'";

    const userPrompt = context
      ? `Context:\n${context}\n\nQuestion: ${query}`
      : `Content: ${fallback || ""}\n\nQuestion: ${query}`;

    const messages = [
      { role: "system", content: systemPrompt },
      { role: "user", content: userPrompt },
    ];

    const chatResponse = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages,
    });

    const reply = chatResponse.choices?.[0]?.message?.content;

    res.status(200).json({
      response: reply || "I can only answer questions related to this website.",
    });
  } catch (error) {
    console.error("Error in /chat:", error);
    res.status(500).json({
      error: error.message || "Something went wrong while processing your request",
    });
  }
});

// Root route
app.get("/", (req, res) => {
  res.send("Assistant Chatbot is running.");
});

// Start the server
const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
});
