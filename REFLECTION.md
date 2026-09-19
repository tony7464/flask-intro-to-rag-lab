# Reflection

This lab made the RAG workflow concrete: the model should not answer until relevant company documents are retrieved and placed into a structured prompt.

Keyword overlap was enough to select different sources for different questions, and the Flask route stayed thin by validating input, calling the service functions, and handling the no-context and model-service failure cases.

The biggest takeaway is that source-backed answers depend more on retrieval and prompt boundaries than on the model itself.
