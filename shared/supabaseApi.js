'use strict';

async function fetchRouteList(sbClient) {
  const { data, error } = await sbClient
    .from('route_files')
    .select('*')
    .order('created_at', { ascending: false });
  if (error) throw new Error(error.message);
  return data;
}

async function downloadGpxFile(sbClient, fileKey) {
  const { data: blob, error } = await sbClient
    .storage.from('gpx_routes')
    .download(fileKey);
  if (error) throw new Error(error.message);
  return blob.text();
}

if (typeof module !== 'undefined') module.exports = { fetchRouteList, downloadGpxFile };
