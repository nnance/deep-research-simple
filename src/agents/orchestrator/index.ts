import type { AgentContext, AgentRequest, AgentResponse } from "@agentuity/sdk";
import { DeepResearchSchema, ResearchSchema } from "../../common/types";

export default async function Agent(
	req: AgentRequest,
	resp: AgentResponse,
	ctx: AgentContext
) {
	const researchParam = DeepResearchSchema.parse(await req.data.json());

	const researcher = await ctx.getAgent({ name: "researcher" });
	if (!researcher) {
		return resp.text("Researcher agent not found", {
			status: 500,
			statusText: "Agent Not Found",
		});
	}

	console.log("Starting research...");
	const researchResults = await researcher.run({ data: researchParam });
	const research = ResearchSchema.parse(await researchResults.data.json());
	console.log("Research completed!");

	const author = await ctx.getAgent({ name: "author" });
	if (!author) {
		return resp.text("Author agent not found", {
			status: 500,
			statusText: "Agent Not Found",
		});
	}
	console.log("Generating report...");
	// Make a copy of research with all properties defined
	const agentResult = await author.run({ data: research });
	const report = await agentResult.data.text();
	console.log("Report generated! report.md");

	return resp.markdown(report);
}
