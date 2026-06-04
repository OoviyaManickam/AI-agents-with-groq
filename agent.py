import os                                                                                                                                                                                
  from groq import Groq                                                                                                                                                                    
                                                                                                                                                                                           
  client = Groq(api_key="your_api_key_here")                                                                                                                                               
   
  messages = [                                                                                                                                                                             
      {"role": "user", "content": "What is an AI agent?"}
  ]                                                                                                                                                                                        
                  
  response = client.chat.completions.create(                                                                                                                                               
      model="llama-3.1-8b-instant",
      messages=messages,                                                                                                                                                                   
      max_tokens=512                                                                                                                                                                       
  )                                                                                                                                                                                        
                                                                                                                                                                                           
  print(response.choices[0].message.content) 