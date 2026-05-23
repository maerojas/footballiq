export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', 'X-Auth-Token');
  if (req.method === 'OPTIONS') return res.status(200).end();
  const endpoint = req.query.endpoint || '';
  const apiKey = process.env.FD_API_KEY || '';
  const url = `https://api.football-data.org/v4/${endpoint}`;
  const response = await fetch(url, { headers: { 'X-Auth-Token': apiKey } });
  const data = await response.json();
  res.status(response.status).json(data);
}
