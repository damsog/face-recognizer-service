from pydantic import BaseModel

class PeerConnectionDTO(BaseModel):
    sdp: str
    sdp_type: str