/**
 * LangChain Story Generator
 *
 * This example demonstrates how to create a multi-step AI workflow using LangGraph
 * to generate a structured story based on user inputs.
 */

import { Annotation, StateGraph } from "@langchain/langgraph";
import { ChatGroq } from "@langchain/groq";
import { z } from "zod";

// ============================================================================
// CONFIGURATION
// ============================================================================

/**
 * Application configuration
 * Adjust these settings to customize the story generation
 */
const CONFIG = {
  sceneCount: 3, // Number of scenes to generate in the story
};

/**
 * Available Groq models you can try:
 * - "gemma2-9b-it"       - Good balance of quality and speed
 * - "llama3-70b-8192"    - High quality for complex stories
 * - "mixtral-8x7b-32768" - Good for longer contexts
 * - "llama3-8b-8192"     - Faster, lower resource usage
 *
 * For the full list of supported models, see:
 * @link https://groq.com/docs/models
 */
const MODEL_NAME = "gemma2-9b-it";

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

/**
 * Define the state schema for our story generation process
 * @link https://js.langchain.com/docs/langgraph/concepts/state
 */
const StoryStateAnnotation = Annotation.Root({
  // User input - what the audience provides to guide the story
  audience_inputs: Annotation<
    {
      category: string; // Type of input (character, setting, plot_twist)
      description: string; // The actual content
    }[]
  >({
    default: () => [],
    reducer: (_current, update) => update,
  }),

  // Story plan - the blueprint for our story
  storyPlan: Annotation<{
    title: string;
    premise: string;
    characters: { name: string; description: string }[];
    settings: { name: string; description: string }[];
    scenes: { title: string; description: string }[];
  }>({
    reducer: (_current, update) => update,
  }),

  // Individual scenes with details
  scenes: Annotation<
    {
      sceneNumber: number;
      title: string;
      description: string;
      characters: string[];
      setting: string;
    }[]
  >({
    default: () => [],
    reducer: (current, update) => current.concat(update),
  }),

  // Final formatted output to return to the user
  finalOutput: Annotation<{
    title: string;
    premise: string;
    scenes: {
      sceneNumber: number;
      title: string;
      description: string;
    }[];
  }>({
    reducer: (_current, update) => update,
  }),
});

// ============================================================================
// MODEL INITIALIZATION
// ============================================================================

/**
 * Initialize the AI model with optimized parameters
 * @link https://js.langchain.com/docs/integrations/chat/groq
 */
const model = new ChatGroq({
  model: MODEL_NAME,
  temperature: 0.5, // Lower for more consistent outputs
  maxTokens: 1500, // Limit token usage per request
  maxRetries: 3, // Retry failed requests
  timeout: 60000, // 60 second timeout
});

// ============================================================================
// PROCESSING NODES
// ============================================================================

/**
 * Process audience inputs - normalizes the input format
 * This handles different ways the inputs might be structured
 */
const processInputs = async (state: typeof StoryStateAnnotation.State) => {
  console.log("Processing user inputs...");

  const inputs = state?.audience_inputs;

  if (Array.isArray(inputs)) {
    return {}; // Already in correct format
  }

  if (inputs && Array.isArray((inputs as { audience_inputs?: unknown })?.audience_inputs)) {
    return { audience_inputs: (inputs as { audience_inputs: unknown }).audience_inputs };
  }

  return {};
};

/**
 * Plan the story based on audience inputs
 * This creates the overall story structure including characters, settings, and scene outlines
 */
const planStory = async (state: typeof StoryStateAnnotation.State) => {
  console.log("Creating story plan...");

  const inputItems = state.audience_inputs || [];

  // Helper function to extract items by category
  const getCategoryItems = (category: string) => {
    return inputItems
      .filter(
        (item) =>
          item.category === category ||
          item.category === category.replace("_", "-")
      )
      .map((item) => item.description);
  };

  // Extract inputs by category
  const characters = getCategoryItems("character");
  const settings = getCategoryItems("setting");
  const plotTwists = getCategoryItems("plot_twist");

  // Define the expected schema for the AI's response
  // @link https://zod.dev/ for Zod schema documentation
  const storyPlanSchema = z.object({
    title: z.string().describe("Title for the murder mystery comic"),
    premise: z.string().describe("Brief premise of the mystery"),
    characters: z.array(
      z.object({
        name: z.string().describe("Character name"),
        description: z.string().describe("Brief character description"),
      })
    ),
    settings: z.array(
      z.object({
        name: z.string().describe("Setting name"),
        description: z.string().describe("Setting description"),
      })
    ),
    scenes: z
      .array(
        z.object({
          title: z.string().describe("Scene title"),
          description: z.string().describe("Brief scene description"),
        })
      )
      .describe(`The story should have ${CONFIG.sceneCount} scenes`),
  });

  // Create a simplified prompt for the AI
  const storyPlanPrompt = `
    Create a brief murder mystery with ${CONFIG.sceneCount} scenes using:
    ${characters.length > 0 ? `CHARACTERS: ${characters.join(", ")}` : "Create 2-3 characters"}
    ${settings.length > 0 ? `SETTINGS: ${settings.join(", ")}` : "Create 1-2 settings"}
    ${plotTwists.length > 0 ? `PLOT TWIST: ${plotTwists[0]}` : ""}
    
    Keep all descriptions under 100 words per scene.
  `;

  // Call the AI with structured output format
  // @link https://js.langchain.com/docs/modules/model_io/output_parsers/structured
  const storyPlan = await model
    .withStructuredOutput(storyPlanSchema, { name: "story_plan" })
    .invoke(storyPlanPrompt);

  // Ensure we have exactly the right number of scenes
  let scenes = storyPlan.scenes;
  if (scenes.length !== CONFIG.sceneCount) {
    console.log(
      `Adjusting scenes from ${scenes.length} to ${CONFIG.sceneCount}`
    );

    // Add scenes if we need more
    while (scenes.length < CONFIG.sceneCount) {
      scenes.push({
        title: `Scene ${scenes.length + 1}`,
        description: `Continuation of the mystery...`,
      });
    }

    // Or trim if we have too many
    scenes = scenes.slice(0, CONFIG.sceneCount);
  }

  return { storyPlan: { ...storyPlan, scenes } };
};

/**
 * Generate detailed scenes based on the story plan
 * Creates each scene with characters, setting, and descriptions
 */
const generateScenes = async (state: typeof StoryStateAnnotation.State) => {
  console.log("Generating individual scenes...");

  const scenes = [];

  // Generate each scene one at a time
  for (let i = 0; i < CONFIG.sceneCount; i++) {
    const sceneNumber = i + 1;
    const scenePlan = state.storyPlan.scenes[i];

    // Define the schema for an individual scene
    const sceneSchema = z.object({
      sceneNumber: z.number().describe(`Scene number (1-${CONFIG.sceneCount})`),
      title: z.string().describe("Scene title"),
      description: z.string().describe("Scene description (max 100 words)"),
      characters: z.array(z.string()).describe("Characters in the scene"),
      setting: z.string().describe("Setting name"),
    });

    // Create a compact prompt for the scene generation
    const scenePrompt = `
      Create scene ${sceneNumber}/${CONFIG.sceneCount} for mystery "${state.storyPlan.title}":
      
      PLAN: ${scenePlan.title}
      
      AVAILABLE CHARACTERS: ${state.storyPlan.characters.map((c) => c.name).join(", ")}
      AVAILABLE SETTINGS: ${state.storyPlan.settings.map((s) => s.name).join(", ")}
      
      Provide:
      1. Brief scene description (max 100 words)
      2. List of 2-3 main characters present
      3. Single setting name
      
      ${sceneNumber === CONFIG.sceneCount ? "Include a cliffhanger ending." : ""}
    `;

    // Generate this specific scene
    const scene = await model
      .withStructuredOutput(sceneSchema, { name: `scene_${sceneNumber}` })
      .invoke(scenePrompt);

    scenes.push(scene);
    console.log(`Generated scene ${sceneNumber}: ${scene.title}`);
  }

  return { scenes };
};

/**
 * Format the final output for presentation to the user
 * Combines all the generated content into a cohesive structure
 */
const formatOutput = async (state: typeof StoryStateAnnotation.State) => {
  console.log("Formatting final output...");

  // Ensure scenes are in the correct order
  const finalScenes = [...state.scenes].sort(
    (a, b) => a.sceneNumber - b.sceneNumber
  );

  // Simplify the scene data for final output
  const simplifiedScenes = finalScenes.map((scene) => ({
    sceneNumber: scene.sceneNumber,
    title: scene.title,
    description: scene.description,
  }));

  return {
    finalOutput: {
      title: state.storyPlan.title,
      premise: state.storyPlan.premise,
      scenes: simplifiedScenes,
    },
  };
};

// ============================================================================
// WORKFLOW DEFINITION
// ============================================================================

/**
 * Create the main workflow graph
 * This defines the sequence of processing steps
 * @link https://js.langchain.com/docs/langgraph/concepts/graph
 */
export const comicBookGraph = new StateGraph(StoryStateAnnotation)
  // Add all processing nodes
  .addNode("processInputs", processInputs)
  .addNode("planStory", planStory)
  .addNode("generateScenes", generateScenes)
  .addNode("formatOutput", formatOutput)

  // Define the flow between nodes
  .addEdge("__start__", "processInputs") // Start with processing inputs
  .addEdge("processInputs", "planStory") // Then plan the story
  .addEdge("planStory", "generateScenes") // Then generate each scene
  .addEdge("generateScenes", "formatOutput") // Then format the final output
  .addEdge("formatOutput", "__end__") // End the process

  // Compile the graph into an executable workflow
  .compile();