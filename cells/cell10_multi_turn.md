# Search Mode 3: See How Citations Change in a Conversation

This is the most advanced search mode. It lets you see how an AI's citations change over the course of a back-and-forth conversation.

### How it Works
Instead of you typing every reply, this cell **simulates a user** for you. Here’s the process:
1.  You provide the **Starting Prompt** to kick off the conversation.
2.  The main AI search agent provides its first response and citations.
3.  Then, a second AI agent, acting as a curious user, reads the response and automatically generates a relevant follow-up question.
4.  This process repeats, creating a natural conversation so you can track how the sources evolve from one turn to the next.

### Key Settings
- **Starting Prompt**: The first message that begins the conversation.
- **Persona**: This is the most powerful feature. You can give the simulated user a personality or a goal. Are they a 'Budget college student' or a 'Travel photographer'? Choosing a persona directs the AI to ask questions relevant to that audience's interests, helping you see what sources are shown to different types of users. You can also select 'Custom' to write your own persona from scratch.
- **Turns**: A "turn" is one complete back-and-forth exchange (one question from the simulated user and one response from the AI assistant). Setting this to '3' will create a three-exchange conversation.
