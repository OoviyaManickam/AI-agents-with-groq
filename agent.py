import os       
  import json                                                                                                                                                                              
  from groq import Groq                                                                                                                                                                    
  from dotenv import load_dotenv                                                                                                                                                           
                                                                                                                                                                                           
  load_dotenv()                                                                                                                                                                            
                                                                                                                                                                                           
  client = Groq(api_key=os.environ.get("GROQ_API_KEY"))                                                                                                                                              
   
  messages = [                                                                                                                                                                             
      {"role": "user", "content": "What is an AI agent?"}
  ]                                                                                                                                                                                        
                  
  response = client.chat.completions.create(                                                                                                                                               
      model="llama-3.1-8b-instant",
      messages=messages,                                                                                                                                                                   
      max_tokens=512                                                                                                                                                                       
  )                                                                                                                                                                                        
                                                                                                                                                                                           
  print(response.choices[0].message.content) 