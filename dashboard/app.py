from fastapi import FastAPI, WebSocket
import asyncio
import uvicorn

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            print(f"[대시보드] 수신: {data}")
            # 여기에 Plotly/Streamlit 등과 연동하여 실시간 시각화 가능
    except Exception as e:
        print(f"[대시보드] 연결 종료: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

