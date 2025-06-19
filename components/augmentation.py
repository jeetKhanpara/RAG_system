from langchain.prompts import PromptTemplate

class Augmentate:
    def __init__(self,retrieved_docs,query):
        self.context = retrieved_docs
        self.query = query

    def create_context(self):
        prompt_context = ''
        for i in self.context:
            prompt_context += i.page_content
            prompt_context += "\n\n"
        return prompt_context

    def create_prompt(self):
        template = PromptTemplate(
                template="""
                You are a helpful assistant.
                Answer ONLY from the provided transcript context.
                If the context is insufficient, just say you don't know.
                keep your answer one liner,
                if you find any irrelevant query then simply answer "out of my business"

                {context}
                Question: {question}
                """,
                input_variables = ['context', 'question']
                )
        prompt = template.invoke({'context':self.create_context(),
                                  'question':self.query})
        return prompt