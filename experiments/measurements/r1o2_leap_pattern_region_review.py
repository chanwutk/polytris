"""Show learned LEAP paths and uncovered endpoints for editing scene regions."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import altair as alt
import duckdb
from PIL import Image
import vl_convert as vlc

from measurements.common import query_dicts


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'measurements-results/r1o2_leap_sampling_diagnostics'
DATASETS = ('amsterdam', 'caldot1', 'caldot2', 'b3d1', 'b3d2', 'b3d3', 'b3d4')
# Separate the few nearby Amsterdam labels without changing the measured points.
LABEL_OFFSETS = [(3, 'E', 40, -22), (5, 'E', 40, 25),
                 (3, 'S', -75, -20), (10, 'E', -75, 0), (21, 'S', -75, 25)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    return parser.parse_args()


def load_patterns(con, metadata):
    # Use the saved learned library, rather than evaluation ground-truth tracks.
    source = RESULTS / f'{metadata["dataset_id"]}-patterns.parquet'
    width, height = metadata['resolution']
    regions = metadata['stop_area']
    assert regions and all(0 <= l < r <= width and 0 <= t < b <= height for l, t, r, b in regions)
    con.execute('CREATE TABLE regions(region INT,l DOUBLE,t DOUBLE,r DOUBLE,b DOUBLE)')
    con.executemany('INSERT INTO regions VALUES (?,?,?,?,?)', [(i, *box) for i, box in enumerate(regions)])
    con.execute('CREATE TABLE points AS SELECT pattern,point,x,y FROM read_parquet(?)', [str(source)])
    assert not con.execute('SELECT pattern,point FROM points GROUP BY ALL HAVING count(*)>1 LIMIT 1').fetchall()
    con.execute('''CREATE TABLE ends AS SELECT pattern,
        first(x ORDER BY point) sx,first(y ORDER BY point) sy,
        last(x ORDER BY point) ex,last(y ORDER BY point) ey
        FROM points GROUP BY pattern''')
    # Match LEAP's inclusive containment and first-matching-region rules exactly.
    con.execute('''CREATE TABLE routes AS SELECT *,
        (SELECT min(region) FROM regions WHERE sx BETWEEN l AND r AND sy BETWEEN t AND b) start_region,
        (SELECT min(region) FROM regions WHERE ex BETWEEN l AND r AND ey BETWEEN t AND b) end_region
        FROM ends''')
    con.execute('''CREATE TABLE endpoints AS
        SELECT pattern,'S' endpoint,sx x,sy y,start_region IS NOT NULL inside FROM routes
        UNION ALL SELECT pattern,'E',ex,ey,end_region IS NOT NULL FROM routes''')
    con.execute('CREATE TABLE offsets(pattern INT,endpoint VARCHAR,dx DOUBLE,dy DOUBLE)')
    if metadata['dataset'] == 'Amsterdam':
        con.executemany('INSERT INTO offsets VALUES (?,?,?,?)', LABEL_OFFSETS)
    # Labels and their leader lines move; the highlighted endpoint coordinates do not.
    labels = query_dicts(con, '''SELECT e.*,concat('P',pattern,'-',endpoint) label,
        greatest(4,least(?-80,x+coalesce(dx,12))) label_x,
        greatest(18,least(?-8,y+coalesce(dy,-12))) label_y
        FROM endpoints e LEFT JOIN offsets USING(pattern,endpoint)
        WHERE NOT inside ORDER BY pattern,endpoint''', [width, height])
    points = query_dicts(con, '''SELECT p.*,start_region IS NOT NULL AND end_region IS NOT NULL eligible
        FROM points p JOIN routes USING(pattern) ORDER BY pattern,point''')
    endpoints = query_dicts(con, 'SELECT * FROM endpoints ORDER BY pattern,endpoint')
    routes = query_dicts(con, 'SELECT * FROM routes ORDER BY pattern')
    total, eligible = con.execute('''SELECT count(*),count(*) FILTER
        (WHERE start_region IS NOT NULL AND end_region IS NOT NULL) FROM routes''').fetchone()
    check = metadata['existing_pattern_check']
    assert (total, eligible) == (check['patterns'], check['eligible_after'])
    assert len(endpoints) == 2 * total
    return source, points, endpoints, labels, routes, total, eligible


def charts(metadata, background, points, endpoints, labels):
    # Keep the frame and every overlay in original image-pixel coordinates.
    width, height = metadata['resolution']
    xscale = alt.Scale(domain=[0, width], nice=False)
    yscale = alt.Scale(domain=[0, height], nice=False, reverse=True)
    x = alt.X('x:Q', scale=xscale, axis=None)
    y = alt.Y('y:Q', scale=yscale, axis=None)
    uri = 'data:image/png;base64,' + base64.b64encode(background.read_bytes()).decode('ascii')
    frame = alt.Chart(alt.Data(values=[dict(url=uri, x=width/2, y=height/2)])).mark_image(
        width=width, height=height).encode(url='url:N', x=x, y=y)
    paths = alt.Chart(alt.Data(values=points)).encode(x=x, y=y, detail='pattern:N', order='point:Q')
    good = paths.transform_filter(alt.datum.eligible).mark_line(color='#11c6a0', opacity=0.55, strokeWidth=3, clip=True)
    bad = paths.transform_filter(~alt.datum.eligible).mark_line(color='#ff9800', opacity=0.95, strokeWidth=3, clip=True)
    ends = alt.Chart(alt.Data(values=endpoints)).mark_point(
        shape='circle', filled=True, fill='white', stroke='#176b63', strokeWidth=1.5,
        size=55, opacity=0.95, clip=True).encode(x=x, y=y)
    uncovered = alt.Chart(alt.Data(values=labels))
    rings = uncovered.mark_point(shape='circle', filled=False, color='#ff3030',
        opacity=1, size=190, strokeWidth=3, clip=True).encode(x=x, y=y)
    leaders = uncovered.mark_rule(color='white', opacity=0.9, strokeWidth=1.2, clip=True).encode(
        x=x, y=y, x2='label_x:Q', y2='label_y:Q')
    text = uncovered.mark_text(align='left', baseline='bottom', fontSize=15,
        fontWeight='bold', color='white', stroke='#111111', strokeWidth=0.6, clip=True).encode(
            x=alt.X('label_x:Q', scale=xscale, axis=None),
            y=alt.Y('label_y:Q', scale=yscale, axis=None), text='label:N')
    boxes = alt.Chart(alt.Data(values=[dict(left=r[0], top=r[1], right=r[2], bottom=r[3])
        for r in metadata['stop_area']])).mark_rect(fill='#30a8ff', fillOpacity=0.025,
            stroke='#30a8ff', strokeWidth=2.5, clip=True).encode(
                x=alt.X('left:Q', scale=xscale, axis=None), x2='right:Q',
                y=alt.Y('top:Q', scale=yscale, axis=None), y2='bottom:Q')
    # Exclude burned-in boxes from the CVAT background so every box remains editable.
    base_layers = [frame, good, bad, ends, leaders, rings, text]
    return {
        'patterns_background': alt.layer(*base_layers).properties(width=width, height=height, padding=0).configure_view(stroke=None),
        'patterns_with_regions': alt.layer(frame, boxes, *base_layers[1:]).properties(width=width, height=height, padding=0).configure_view(stroke=None),
    }


def export_cvat(slug, metadata, output, template):
    # Reuse the XML structure already imported successfully for the original task.
    root = ET.fromstring(template)
    root.find('meta/task/name').text = metadata['dataset'] + ' LEAP pattern region review'
    image = root.find('image')
    image.clear()
    width, height = metadata['resolution']
    name = f'{slug}_leap_patterns.png'
    image.attrib.update(id='0', name=name, width=str(width), height=str(height))
    for left, top, right, bottom in metadata['stop_area']:
        ET.SubElement(image, 'box', label='region', source='manual', occluded='0',
            xtl=str(left), ytl=str(top), xbr=str(right), ybr=str(bottom), z_order='0')
    ET.indent(root, space='  ')
    xml = ET.tostring(root, encoding='utf-8', xml_declaration=True)
    (output/'annotations.xml').write_bytes(xml)
    package = output/f'{slug}-leap-pattern-regions.zip'
    with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('annotations.xml', xml)
        archive.write(output/'patterns_background.png', 'images/'+name)
    # Verify exact coordinates and the task image name inside the uploadable archive.
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {'annotations.xml', 'images/'+name}
        parsed = ET.fromstring(archive.read('annotations.xml')).find('image')
        assert parsed.attrib == image.attrib
        assert [[float(b.get(k)) for k in ('xtl', 'ytl', 'xbr', 'ybr')] for b in parsed] == metadata['stop_area']
        assert archive.read('images/'+name) == (output/'patterns_background.png').read_bytes()


def main():
    args = parse_args()
    frames = RESULTS/'region_frames'
    template = (frames/'amsterdam_cvat/annotations.xml').read_bytes()
    alt.data_transformers.disable_max_rows()
    for slug in args.datasets:
        current = frames/f'{slug}_cvat/edited_regions'
        metadata = json.loads((current/'regions.json').read_text())
        output = current/'pattern_review'
        assert not output.exists(), f'Refusing to overwrite {output}'
        background = frames/f'{metadata["dataset_id"]}-frame.png'
        with Image.open(background) as image:
            assert list(image.size) == metadata['resolution']
        with duckdb.connect() as con:
            source, points, endpoints, labels, routes, total, eligible = load_patterns(con, metadata)
        output.mkdir()
        for name, chart in charts(metadata, background, points, endpoints, labels).items():
            vector = vlc.vegalite_to_svg(chart.to_json())
            (output/f'{name}.svg').write_text(vector)
            (output/f'{name}.png').write_bytes(vlc.svg_to_png(vector))
            with Image.open(output/f'{name}.png') as image:
                image.load()
                assert list(image.size) == metadata['resolution']
        export_cvat(slug, metadata, output, template)
        report = dict(dataset=metadata['dataset'], patterns=total, eligible=eligible,
            uncovered_endpoints=labels, patterns_detail=routes, stop_area=metadata['stop_area'],
            source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            image_name=f'{slug}_leap_patterns.png', resolution=metadata['resolution'])
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        (output/'README.txt').write_text(
            'LEAP learned motion-pattern review\n\n'
            'Teal paths: eligible. Orange paths: ineligible under the supplied regions.\n'
            'White circles: pattern endpoints. Red rings: endpoints outside all supplied boxes.\n'
            'P5-S / P5-E label pattern 5 start/end; pattern IDs are zero-based.\n'
            'Blue rectangles in patterns_with_regions.png show the current stop_area.\n\n'
            'Import the ZIP into a new empty CVAT project as CVAT 1.1.\n'
            'Its image contains the paths and endpoint markers; its region boxes remain editable.\n'
            'Move, resize, remove, or add boxes using label region. Save and export CVAT for images 1.1 without images.\n'
            'The CVAT background is static: colors and red rings do not update while boxes are edited.\n'
            'Some paths may be fragments ending mid-road; region eligibility alone does not establish path quality.\n'
            'No extraction, training, or accuracy evaluation was rerun. Live CVAT import was not tested.\n')
        print(json.dumps(dict(dataset=metadata['dataset'], patterns=total, eligible=eligible,
            uncovered_endpoints=len(labels), output=str(output))), flush=True)


if __name__ == '__main__':
    main()
